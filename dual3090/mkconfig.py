#!/usr/bin/env python3
"""Turn the config `setup.py` recorded into the dual-3090 peer config.

Reads the config the upstream setup wrote (/data/config/strata-<model>.json in
the container) and writes a tuned copy for the two-card peer build:

  * the second card becomes a peer expert tier + prefill helper (--peer-device)
  * the prefill fast paths of this fork are switched on (env STRATA_GR_V3,
    STRATA_PF_GDN_SPLIT, STRATA_PREFILL_STREAM_MIN)
  * expert misses are answered by the CPU, not by DMA over PCIe (--pcie-frac 0)
  * the conversation cache keeps switched-away chats warm
  * the image encoder moves to the helper card and gets 2,048 tokens a picture
  * official sampling (temperature 1.0, top_p 0.95, top_k 20) is set when absent

Machine-shaped values adapt themselves: --ple-io ram and the big conversation
cache are only written when the host has the RAM for it (from /proc/meminfo),
and --peer-reserve-mib follows the quant. Flags you already gave keep their
value, so an existing --max-context or --kv is respected.

Usage:  mkconfig.py <setup-config.json> -o <out.json> [--gate single|dual]
                    [--helper-device N] [--high-ram on|off|auto]
"""
import argparse
import json
import re
import sys
from pathlib import Path

# MiB left free on the helper card: the vision encoder (~1.34 GB) warms up
# there, plus a safety margin (under ~250 MiB free the first prompt OOMs).
PEER_RESERVE = {"iq3_xxs": "1850", "iq3_s": "1850", "iq2_xs": "2700", "q2_0": "2700", "iq1_m": "1850"}
# The n-gram table is mlocked (--ple-io ram) from this much RAM up; below it
# the default SSD reads cost a few seconds on cold prompts but nothing else.
PLE_RAM_MIN_GIB = 96
CONV_CACHE_GIB = [(96, "12288"), (72, "8192")]  # (min host GiB, --conversation-cache-mib)

GATE_ENV = {"STRATA_IQ_MT_MIN": "1", "STRATA_MMQ_NO_STREAMK": "1"}


def host_mem_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0  # not Linux (e.g. editing on a Mac): treat as unknown, use safe defaults


def split_args(args):
    """flat flag list -> ordered dict {flag: value-or-None}"""
    out, i = {}, 0
    while i < len(args):
        if args[i].startswith("--"):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                out[args[i]] = args[i + 1]
                i += 2
            else:
                out[args[i]] = None
                i += 1
        else:
            raise SystemExit(f"mkconfig: argument {args[i]!r} without a flag before it")
    return out


def join_args(flags):
    out = []
    for k, v in flags.items():
        out.append(k)
        if v is not None:
            out.append(str(v))
    return out


def model_tag(cfg):
    m = re.search(r"-(q2_0|iq2_xs|iq3_xxs|iq3_s|iq1_m)$", cfg.get("model_name", ""))
    if m:
        return m.group(1)
    for t in PEER_RESERVE:  # fall back to a path mention
        if f"/{t}" in json.dumps(cfg.get("args", [])).lower() or f"{t}/" in json.dumps(cfg.get("args", [])).lower():
            return t
    raise SystemExit("mkconfig: cannot tell the quant (q2_0/iq2_xs/iq3_xxs/iq3_s/iq1_m) from the config")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the JSON config that setup.py recorded")
    ap.add_argument("-o", "--out", required=True, help="where to write the dual config")
    ap.add_argument("--gate", choices=("single", "dual"),
                    help="write an exactness-gate config instead (single = one card reference, "
                         "dual = the peer build with a fixed expert set)")
    ap.add_argument("--helper-device", type=int, default=None,
                    help="nvidia-smi number of the second card (the one without the desktop). "
                         "Default: the engine sees both cards in nvidia-smi order, so the helper is 1.")
    ap.add_argument("--high-ram", choices=("on", "off", "auto"), default="auto",
                    help="mlock the n-gram table (--ple-io ram); auto = from /proc/meminfo")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    flags = split_args(cfg["args"])
    tag = model_tag(cfg)
    mem = host_mem_gib()
    changes = []

    if a.gate == "single":
        # The one-card reference: no peer, no split kernels, a fixed CPU expert
        # set, no swapping during the gate. Runs with the gate env below.
        for drop in ("--peer-device", "--peer-reserve-mib", "--peer-slots", "--peer-adapt-swaps"):
            flags.pop(drop, None)
        flags["--expert-cache"] = "6000"
        flags["--adapt-swaps"] = "0"
        flags["--pcie-frac"] = "0"
    else:
        # the peer tier: the second card holds a second expert tier and computes
        # part of every prompt (QSA split + GDN split, both exact)
        flags["--peer-device"] = "1"
        flags["--peer-reserve-mib"] = PEER_RESERVE[tag]
        flags["--expert-cache"] = flags.get("--expert-cache", "auto")
        flags["--pcie-frac"] = "0"     # both cards nearly full: misses go to the CPU
        flags["--mtp-window"] = "4096"
        if a.gate == "dual":
            # the same expert set as the single reference (gate.sh explains)
            flags["--expert-cache"] = "3000"
            flags["--adapt-swaps"] = "0"
            flags["--peer-slots"] = "3950"
            flags["--peer-adapt-swaps"] = "0"
        else:
            eff_mem = 1024.0 if a.high_ram == "on" else mem
            gib = next((v for need, v in CONV_CACHE_GIB if eff_mem >= need), None)
            if gib:
                flags["--conversation-cache-mib"] = gib
                flags["--conversation-cache-slots"] = "4"
                flags["--conversation-cache-min-tokens"] = "2048"
        if not a.gate and (a.high_ram == "on" or (a.high_ram == "auto" and mem >= PLE_RAM_MIN_GIB)):
            flags["--ple-io"] = "ram"  # mmap + mlock the 28.8 GB n-gram table
        elif not a.gate:
            flags.pop("--ple-io", None)

    if a.gate:
        # the gate is a text-only comparison (the measured gate configs carry no
        # vision); the encoder's ~1.3 GB on the helper card would change the
        # peer's auto-sized buffers.
        flags.pop("--vision", None)
        flags.pop("--vram-reserve-mib", None)
        cfg.pop("vision", None)
    elif cfg.get("vision"):
        # the setup's own --vram-reserve-mib keeps VRAM for the encoder ON THE
        # ENGINE card; here the encoder runs on the helper instead, and the
        # helper's headroom is --peer-reserve-mib's job.
        flags.pop("--vram-reserve-mib", None)

    cfg["args"] = join_args(flags)

    # the setup records which single card to run on; the peer build takes both.
    for k in ("gpu", "gpus_asked", "layer_split"):
        if cfg.pop(k, None) is not None:
            changes.append(f"removed {k!r} (the peer build runs on both cards)")

    env = dict(cfg.get("env") or {})
    if a.gate == "single":
        pass  # the reference arm stays bare, exactly like the recorded reference
    else:
        env.setdefault("STRATA_GR_V3", "1")         # hc-read kernels (measured, exact)
        env.setdefault("STRATA_PF_GDN_SPLIT", "1")  # GDN heads split over NVLink (+6-8 % prompt)
        env.setdefault("STRATA_PREFILL_STREAM_MIN", "1024")
    if a.gate:
        # the deterministic pair: the exactness switches (#152) + no stream-k
        # tiling, so single and dual run the same kernels bit for bit; the dual
        # arm keeps the tuning env above, a PASS proves the serving path exact.
        env.update(GATE_ENV)
    cfg["env"] = env

    if not cfg.get("sampling"):
        cfg["sampling"] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
        changes.append("sampling set to the model card's official values")

    if cfg.get("vision"):
        cfg["vision"]["max_tokens"] = 2048
        helper = a.helper_device if a.helper_device is not None else 1
        cfg["vision"]["cuda_device"] = helper
        changes.append(f"image encoder -> card {helper}, 2048 tokens a picture")

    if a.gate:
        cfg["model_name"] = cfg.get("model_name", "strata") + f"-gate-{a.gate}"

    out = Path(a.out)
    out.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"mkconfig: wrote {out}")
    print(f"  quant {tag}, host RAM {mem:.0f} GiB, --ple-io {flags.get('--ple-io', 'default (SSD)')},"
          f" peer reserve {flags.get('--peer-reserve-mib', '-')} MiB")
    for c in changes:
        print("  " + c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
