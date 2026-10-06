#!/usr/bin/env python3
"""flashbench - one benchmark for every OpenAI-compatible LLM server (vLLM, Strata, llama.cpp, ...).

Measures PREFILL (prompt reading) and DECODE (writing) speed with the SAME requests on every engine, so
numbers from different engines and different days can be compared directly.  Pure Python standard
library; runs on the PC or on the Mac.  Full user guide: README.md next to this file.

  python3 flashbench.py run --url http://127.0.0.1:8000/v1 --label vllm-live
  python3 flashbench.py run --url http://127.0.0.1:8080/v1 --label strata-iq3s --quick
  python3 flashbench.py compare results/*.json --md table.md --png chart.png
  python3 flashbench.py gate --url ... --save ref.json        (exact-output reference, greedy)
  python3 flashbench.py gate --url ... --against ref.json     (is the output still identical?)
  python3 flashbench.py calibrate --url <vLLM /v1>            (maintainer only: rebuild cuts.json)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import http.client
import json
import os
import secrets
import statistics
import sys
import time
import urllib.parse
import urllib.request

VERSION = "1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.txt")
CUTS = os.path.join(HERE, "cuts.json")

HEAD = "You are an expert C/C++ reviewer. The source code to review follows.\n\n<source>\n"
QUESTION = ("\n</source>\n\nWrite a long, detailed code review of the source above: explain what the code "
            "does, then list concrete bugs, risks and improvements, each with a short code example. "
            "Write at least 1500 words and do not stop early.")
MARK_LEN = 16                                   # hex chars of the per-request random id

MODES = {
    "greedy": {"temperature": 0.0},
    "sampled": {"temperature": float(os.environ.get("FB_TEMP", "0.7")), "top_p": 0.95, "top_k": 20},  # FB_TEMP=1.0: the live config
}
DEFAULT_CONTEXTS = "1k,8k,32k,64k,128k"


# ------------------------------------------------------------------------------------------ helpers
def parse_ctx(s: str) -> int:
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


def ctx_label(n: int) -> str:
    return f"{n // 1000}K" if n % 1000 == 0 else str(n)


def corpus() -> str:
    with open(CORPUS, encoding="utf-8") as f:
        return f.read()


def user_text(chars: int, mark: str) -> str:
    return f"[bench-id {mark}]\n" + HEAD + corpus()[:chars] + QUESTION


def load_cuts() -> dict:
    if os.path.exists(CUTS):
        with open(CUTS) as f:
            return json.load(f)
    return {}


def chars_for(target: int, cuts: dict) -> int:
    """Characters of corpus giving `target` prompt tokens (exact table from calibrate, else interpolated)."""
    table = {int(k): v for k, v in cuts.get("cuts", {}).items()}
    if target in table:
        return table[target]
    cpt = cuts.get("chars_per_token", 3.3)
    overhead = cuts.get("overhead_tokens", 90)
    return max(0, int((target - overhead) * cpt))


def headers(api_key: str | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def get_json(url: str, api_key=None, timeout=10):
    req = urllib.request.Request(url, headers=headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_model(base: str, api_key=None) -> tuple[str, dict]:
    info = get_json(base.rstrip("/") + "/models", api_key)
    return info["data"][0]["id"], info


def stream_chat(base: str, model: str, text: str, max_tokens: int, sampling: dict, api_key=None,
                timeout=7200, seed=1234) -> dict:
    """One streamed chat request.  Returns timings measured on the client side."""
    u = urllib.parse.urlparse(base.rstrip("/") + "/chat/completions")
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}, **sampling}
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", u.path, json.dumps(body), headers(api_key))
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    t_first = t_last = None
    chunks, usage, finish, out = 0, None, None, []
    buf = b""
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue                              # blank lines, SSE comments / keep-alives
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            ev = json.loads(data)
        except ValueError:
            buf += data
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
            if piece or d.get("tool_calls"):
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
                chunks += 1
                out.append(piece)
            finish = ch.get("finish_reason") or finish
    t_end = time.perf_counter()
    conn.close()
    if t_first is None:
        raise RuntimeError("the server streamed no tokens")
    n_out = (usage or {}).get("completion_tokens") or chunks
    n_in = (usage or {}).get("prompt_tokens")
    ttft = t_first - t0
    span = t_last - t_first
    return {"prompt_tokens": n_in, "completion_tokens": n_out, "ttft_s": round(ttft, 3),
            "prefill_tps": round(n_in / ttft, 1) if n_in and ttft > 0 else None,
            "decode_s": round(span, 3), "decode_tps": round((n_out - 1) / span, 2) if span > 0 and n_out > 1 else None,
            "e2e_s": round(t_end - t0, 3), "finish": finish, "stream_chunks": chunks, "text": "".join(out)}


# ------------------------------------------------------------------------------------------ run
def cmd_run(a) -> int:
    model, info = (a.model, {}) if a.model else server_model(a.url, a.api_key)
    cuts = load_cuts()
    contexts = [parse_ctx(c) for c in a.contexts.split(",")]
    modes = a.modes.split(",")
    for m in modes:
        if m not in MODES:
            sys.exit(f"unknown mode {m!r}; known: {', '.join(MODES)}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{stamp}_{a.label}.json")
    res = {"tool": "flashbench", "version": VERSION, "label": a.label, "note": a.note, "url": a.url,
           "model": model, "server_models": info, "started": dt.datetime.now().isoformat(timespec="seconds"),
           "settings": {"contexts": contexts, "modes": modes, "gen_tokens": a.gen, "repeats": a.repeats,
                        "long_repeats": a.long_repeats, "warmup": not a.no_warmup},
           "runs": []}

    def save():
        res["summary"] = summarize(res["runs"])
        with open(path, "w") as f:
            json.dump(res, f, indent=1)

    print(f"flashbench {VERSION}: {a.label}  model={model}  -> {path}", flush=True)
    if not a.no_warmup:
        print("  warm-up (not counted) ...", flush=True)
        try:
            r = stream_chat(a.url, model, user_text(chars_for(1000, cuts), secrets.token_hex(MARK_LEN // 2)),
                            128, MODES["greedy"], a.api_key, a.timeout)
            print(f"    ok: {r['prompt_tokens']} in / {r['completion_tokens']} out", flush=True)
        except Exception as e:                    # noqa: BLE001
            print(f"    warm-up failed: {e}", flush=True)
            return 2
    for ctx in contexts:
        reps = a.long_repeats if ctx >= 100_000 else a.repeats
        for mode in modes:
            for rep in range(1, reps + 1):
                mark = secrets.token_hex(MARK_LEN // 2)
                text = user_text(chars_for(ctx, cuts), mark)
                try:
                    r = stream_chat(a.url, model, text, a.gen, MODES[mode], a.api_key, a.timeout)
                    err = None
                except Exception as e:            # noqa: BLE001
                    r, err = {}, str(e)
                row = {"ctx": ctx, "mode": mode, "rep": rep, "mark": mark, "error": err,
                       **{k: v for k, v in r.items() if k != "text"}}
                if a.keep_text:
                    row["text"] = r.get("text")
                res["runs"].append(row)
                save()
                if err:
                    print(f"  {ctx_label(ctx):>5} {mode:8} #{rep}: ERROR {err}", flush=True)
                else:
                    short = " (SHORT answer)" if r["completion_tokens"] < a.gen * 0.5 else ""
                    print(f"  {ctx_label(ctx):>5} {mode:8} #{rep}: prompt {r['prompt_tokens']:>6} tok  "
                          f"prefill {r['prefill_tps'] or 0:>7.1f} tok/s  decode {r['decode_tps'] or 0:>6.1f} tok/s  "
                          f"({r['completion_tokens']} out, {r['finish']}){short}", flush=True)
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    save()
    print()
    print(markdown([res]))
    print(f"\nsaved: {path}")
    return 0


def summarize(runs):
    out = {}
    for r in runs:
        if r.get("error"):
            continue
        key = f"{r['ctx']}|{r['mode']}"
        d = out.setdefault(key, {"ctx": r["ctx"], "mode": r["mode"], "prefill": [], "decode": [], "prompt_tokens": []})
        if r.get("prefill_tps"):
            d["prefill"].append(r["prefill_tps"])
        if r.get("decode_tps"):
            d["decode"].append(r["decode_tps"])
        d["prompt_tokens"].append(r.get("prompt_tokens"))
    for d in out.values():
        d["prefill_median"] = round(statistics.median(d["prefill"]), 1) if d["prefill"] else None
        d["decode_median"] = round(statistics.median(d["decode"]), 1) if d["decode"] else None
    return out


# ------------------------------------------------------------------------------------------ compare
def load_results(paths):
    res = []
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            with open(q) as f:
                r = json.load(f)
            r["summary"] = summarize(r["runs"])
            res.append(r)
    return res


def markdown(results) -> str:
    ctxs = sorted({d["ctx"] for r in results for d in r["summary"].values()})
    modes = [m for m in MODES if any(d["mode"] == m for r in results for d in r["summary"].values())]
    lines = []

    def table(title, metric, mode_list):
        lines.append(f"**{title}** (tokens/s, median)\n")
        lines.append("| engine | mode | " + " | ".join(ctx_label(c) for c in ctxs) + " |")
        lines.append("|---|---|" + "---:|" * len(ctxs))
        for r in results:
            for m in mode_list:
                cells = []
                for c in ctxs:
                    d = r["summary"].get(f"{c}|{m}")
                    v = d and d.get(metric)
                    cells.append(f"{v:.0f}" if metric == "prefill_median" and v else f"{v:.1f}" if v else "-")
                if any(x != "-" for x in cells):
                    lines.append(f"| {r['label']} | {m} | " + " | ".join(cells) + " |")
        lines.append("")

    table("Prefill - prompt reading", "prefill_median", modes)
    table("Decode - writing", "decode_median", modes)
    return "\n".join(lines)


def cmd_compare(a) -> int:
    results = load_results(a.files)
    md = markdown(results)
    print(md)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md + "\n")
    if a.png:
        chart(results, a.png, a.mode)
        print(f"chart: {a.png}")
    return 0


def chart(results, path, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctxs = sorted({d["ctx"] for r in results for d in r["summary"].values()})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    w = 0.8 / max(1, len(results))
    for ax, metric, title in ((axes[0], "prefill_median", "Prefill (prompt reading)"),
                              (axes[1], "decode_median", f"Decode (writing), {mode}")):
        for i, r in enumerate(results):
            m = "greedy" if metric == "prefill_median" else mode
            vals = [(r["summary"].get(f"{c}|{m}") or {}).get(metric) or 0 for c in ctxs]
            xs = [j + (i - (len(results) - 1) / 2) * w for j in range(len(ctxs))]
            bars = ax.bar(xs, vals, w, label=r["label"])
            for x, v in zip(xs, vals):
                if v:
                    ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(ctxs)), [ctx_label(c) for c in ctxs])
        ax.set_xlabel("prompt length (tokens)")
        ax.set_ylabel("tokens/s")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ------------------------------------------------------------------------------------------ gate
GATE_PROMPTS = [
    ("chat", "Explain in plain words how a refrigerator works, step by step."),
    ("code", None),                                # the corpus cut to ~4K tokens, fixed id
    ("math", "A train leaves at 09:40 and travels 212 km at 83 km/h. When does it arrive? Show the working."),
]


def cmd_gate(a) -> int:
    model = a.model or server_model(a.url, a.api_key)[0]
    cuts = load_cuts()
    outs = {}
    for name, text in GATE_PROMPTS:
        t = text or user_text(chars_for(4000, cuts), "0" * MARK_LEN)
        r = stream_chat(a.url, model, t, a.gen, MODES["greedy"], a.api_key, a.timeout)
        outs[name] = r["text"]
        print(f"  {name}: {r['completion_tokens']} tokens, decode {r['decode_tps']} tok/s", flush=True)
    if a.save:
        with open(a.save, "w") as f:
            json.dump({"model": model, "url": a.url, "gen": a.gen, "outputs": outs}, f, indent=1)
        print(f"reference saved: {a.save}")
    if a.against:
        with open(a.against) as f:
            ref = json.load(f)["outputs"]
        bad = 0
        for name, got in outs.items():
            exp = ref.get(name, "")
            if got == exp:
                print(f"  {name}: IDENTICAL")
            else:
                bad += 1
                i = next((k for k, (x, y) in enumerate(zip(got, exp)) if x != y), min(len(got), len(exp)))
                print(f"  {name}: DIFFERS at char {i}: ref={exp[max(0, i-40):i+40]!r}\n"
                      f"  {' ' * len(name)}           got={got[max(0, i-40):i+40]!r}")
        print("GATE: PASS" if not bad else f"GATE: FAIL ({bad} of {len(outs)} differ)")
        return 1 if bad else 0
    return 0


# ------------------------------------------------------------------------------------------ calibrate
def tokenize_count(base: str, model: str, text: str) -> int:
    root = base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    body = {"model": model, "messages": [{"role": "user", "content": text}], "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(root + "/tokenize", json.dumps(body).encode(), headers())
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["count"]


def cmd_calibrate(a) -> int:
    """Find, for each target, the corpus cut whose prompt tokenizes to exactly that many tokens (needs a
    server with vLLM's /tokenize).  Only needed if corpus.txt or the prompt wording ever changes."""
    model = a.model or server_model(a.url)[0]
    mark = "0" * MARK_LEN
    base_tokens = tokenize_count(a.url, model, user_text(0, mark))
    total = len(corpus())
    cuts = {}
    for target in [parse_ctx(c) for c in a.targets.split(",")]:
        lo, hi = 0, total
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tokenize_count(a.url, model, user_text(mid, mark)) <= target:
                lo = mid
            else:
                hi = mid - 1
        cuts[str(target)] = lo
        print(f"  {ctx_label(target)}: {lo} chars", flush=True)
    big = max(cuts, key=int)
    cpt = cuts[big] / (int(big) - base_tokens)
    out = {"model": model, "overhead_tokens": base_tokens, "chars_per_token": round(cpt, 4), "cuts": cuts,
           "made": dt.date.today().isoformat()}
    with open(CUTS, "w") as f:
        json.dump(out, f, indent=1)
    print(f"written {CUTS}")
    return 0


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure prefill + decode")
    r.add_argument("--url", required=True, help="OpenAI base URL, e.g. http://127.0.0.1:8000/v1")
    r.add_argument("--label", required=True, help="a short name for this engine/setup (goes into the file name)")
    r.add_argument("--note", default="", help="free text stored with the results (settings, versions ...)")
    r.add_argument("--model", help="model name to send (default: the first one the server lists)")
    r.add_argument("--api-key")
    r.add_argument("--contexts", default=DEFAULT_CONTEXTS, help=f"prompt lengths (default {DEFAULT_CONTEXTS})")
    r.add_argument("--modes", default="greedy,sampled", help="greedy, sampled or both (default both)")
    r.add_argument("--gen", type=int, default=512, help="tokens to generate per request (default 512)")
    r.add_argument("--repeats", type=int, default=3, help="runs per length and mode (default 3)")
    r.add_argument("--long-repeats", type=int, default=2, help="runs for lengths >= 100K (default 2)")
    r.add_argument("--quick", action="store_true", help="short check: 1K + 32K, greedy, 1 run")
    r.add_argument("--no-warmup", action="store_true")
    r.add_argument("--keep-text", action="store_true", help="store the generated text in the results")
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--outdir", default=os.path.join(HERE, "results"))

    c = sub.add_parser("compare", help="table (+ chart) from result files")
    c.add_argument("files", nargs="+")
    c.add_argument("--md", help="also write the table to this markdown file")
    c.add_argument("--png", help="also draw a chart (needs matplotlib)")
    c.add_argument("--mode", default="greedy", help="decode mode shown in the chart (default greedy)")

    g = sub.add_parser("gate", help="exact-output check (greedy) against a saved reference")
    g.add_argument("--url", required=True)
    g.add_argument("--model")
    g.add_argument("--api-key")
    g.add_argument("--gen", type=int, default=256)
    g.add_argument("--timeout", type=int, default=1800)
    g.add_argument("--save", help="write the outputs as the new reference")
    g.add_argument("--against", help="compare with this reference")

    k = sub.add_parser("calibrate", help="maintainer: rebuild cuts.json (needs vLLM's /tokenize)")
    k.add_argument("--url", required=True)
    k.add_argument("--model")
    k.add_argument("--targets", default="1k,4k,8k,16k,32k,64k,100k,128k,200k")

    a = ap.parse_args()
    if a.cmd == "run" and a.quick:
        a.contexts, a.modes, a.repeats, a.long_repeats = "1k,32k", "greedy", 1, 1
    return {"run": cmd_run, "compare": cmd_compare, "gate": cmd_gate, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
