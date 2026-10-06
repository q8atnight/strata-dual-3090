# Strata on two RTX 3090s — the dual-3090 build (SF3.7)

[Strata](https://github.com/Niko1221/Strata) runs Qwen3.8-Flash-Next (a large
mixture-of-experts model as a GGUF) on one big graphics card. This fork makes **two RTX 3090s
work as one engine** for a single chat: the second card holds a second tier of experts and
computes part of every prompt, over the NVLink bridge.

It is Strata **v0.1.38 plus 29 commits** (see [Provenance](#provenance)) and it is **exact**:
the two-card answers are word for word the one-card answers —
[`gate.sh`](verify/gate.sh) proves it on your machine.

## What it does, and how that differs from the built-in layer split

Upstream's own multi-GPU mode splits the *layers* across cards
([docs/MULTI_GPU.md](../docs/MULTI_GPU.md)). This build keeps the whole model on the primary
card and adds the second card as a **peer**:

- a **second expert tier** — the experts the primary card cannot keep resident live on the
  helper's VRAM and are read over NVLink inside the decode window (a dual 3090 pair holds
  ~19,500 of the 24,576 IQ3_S experts; hit rate 0.98 on real chats);
- **prompt splitting** — the helper computes the selection and attention of half of every
  prompt's sparse-attention queries and a share of the GDN recurrence, results crossing
  NVLink while the primary works on its half;
- the **image encoder** runs on the helper card, so vision costs the engine no experts.

One conversation at a time: this is a single-stream build. A second request waits.

## Numbers

flashbench ([`verify/flashbench.py`](verify/flashbench.py)), IQ3_S, 262K context, greedy,
median of 3 — measured on a Ryzen 9 3950X, 121 GB RAM, 2× RTX 3090 (NVLink), CUDA 13:

| build | reads prompt (prefill) 8K / 32K / 128K | writes answer (decode) 8K / 32K / 128K |
|---|---:|---:|
| stock Strata 0.1.13, one 3090 | 1,376 / 1,361 / 1,223 | 50.7 / 46.2 / 41.2 |
| vLLM, both 3090s (0.1.13-era comparison) | 2,000 / 2,219 / 2,204 | 80.9 / 80.8 / 81.3 |
| **this build (SF3.7), both 3090s** | **2,908 / 3,231 / 3,186** | **116.0 / 113.0 / 111.3** |

Sampled (the live settings, temperature 1.0): 102-104 t/s decode, a 1.5K prompt reads at
1,672 t/s, and a chat you switched back to is warm again in ~1.1 s (conversation cache).
Ten verifiable quality questions came back 10/10, and the gate says the words are exactly
what one card would have written.

Roughly: the second card adds ~+35 % to prompt reading and ~+10 % to decode at a single chat
(the decode window is latency-bound on the primary card either way). Its biggest job is
expert residency — without it most prompts miss experts and pay RAM prices.

## Requirements

- **2× RTX 3090 + an NVLink bridge.** The peer tier reads expert rows over NVLink; consumer
  cards have no usable PCIe peer-to-peer.
- Linux, NVIDIA driver ≥ 580, `nvidia-container-toolkit`. The CUDA version on the host does
  not matter — the compiler lives inside the image (sm_86; rebuild with `ARCHS=` for others).
- RAM: **IQ3_S wants 64 GB** with little else running (96+ GB also unlocks the mlocked
  n-gram table and the big conversation cache, chosen automatically); **Q2_0 / IQ2_XS run
  from 48 GB**.
- Disk: ~150 GB free (the model downloads are 66-84 GB; the prepared pack adds more).
- A modern 16-thread CPU helps: the CPU answers expert misses and runs the n-gram drafter.

## Quick start

```sh
git clone https://github.com/q8atnight/strata-dual-3090.git
cd strata-dual-3090
./dual3090/build.sh        # compiles engine + image encoder into an image (a few minutes)
./dual3090/start.sh        # first start downloads the model (~70-90 GB) and prepares the pack
```

Then: web chat at `http://<this PC>:8080/`, OpenAI- and Anthropic-compatible API at
`http://<this PC>:8080/v1` (model `qwen3.8-flash-next-iq3_s`). Stop with `./dual3090/stop.sh`.
`/health` answers `"loaded": true` and `"images": true` when everything is up — an engine
that died during loading keeps answering `"loaded": false` while requests hang, so check it
when the model "does not reply". `docker logs -f strata-dual3090` has the reason; the usual
ones are VRAM (another program grabbed a card) or RAM.

Settings are environment variables in front of `start.sh`:

| variable | meaning |
|---|---|
| `MODEL=IQ3_S\|IQ2_XS\|Q2_0` | the quant: Q2_0 fastest, IQ3_S best quality |
| `CONTEXT=262144` | served context (default 262K) |
| `VISION=yes\|no\|cpu` | image encoder, default on the helper card |
| `GPUS_ORDER=1,0` | on a PC with a monitor: put the card **without** the desktop first |
| `API_KEY=*** `ALLOWED_HOSTS="name ip"` | before exposing the port beyond this PC |
| `RECONFIG=1` / `REINSTALL=1` | rewrite the dual config / re-run the model setup |

First start on a fresh machine takes a while (the download dominates). Everything lives on
the `strata-dual-data` Docker volume; `docker volume rm strata-dual-data` is the full reset.

## Verify it on your box

```sh
./dual3090/verify/verify.sh quick   # ~1 minute: does it reach the numbers above?
./dual3090/verify/gate.sh           # two stages, ~5 min each: dual answers == single
                                    #   answers, and your single vs a reference recorded
                                    #   from stock v0.1.38
```

The gate runs both arms with a fixed expert set and the exactness switches
(`STRATA_IQ_MT_MIN=1`, `STRATA_MMQ_NO_STREAMK=1`, upstream #152) so they execute the same
kernels; a PASS means the peer path added nothing to the words. `verify/longgate.py` does
the same with 19K-token prompts if you want the deeper check (`save` then `check`).

## What the tuning actually changes

Everything this fork adds is exact by construction or was A/B-gated here; the numbers below
are ours, measured on the box above. [`mkconfig.py`](mkconfig.py) writes these into your
config — this is also what to keep if you hand-edit:

- `--peer-device 1` — the second card as expert tier + prompt helper. `--peer-reserve-mib`
  keeps its headroom for the image encoder (1,850 MiB on IQ3_S, 2,700 on the 2-bit quants);
  under ~250 MiB free the first prompt dies "prefill mmq: quantize: out of memory".
- `STRATA_PF_GDN_SPLIT=1` — GDN recurrence split over NVLink: +6-8 % prefill, decode ±0.
- `STRATA_GR_V3=1` — the hyperconnection read as two kernels instead of many small ones.
- `STRATA_PREFILL_STREAM_MIN=1024` — stream prompts from 1,024 tokens (measured floor here).
- `--pcie-frac 0` — expert misses go through the CPU: with both cards nearly full, DMA of
  misses made every verify window wait, dropping it won +13 % decode.
- `--mtp-window 4096`; `--kv int8 --kv-resident 32768` (KV streaming above 64K context);
  conversation cache (12 GiB on 96 GB+ machines, 8 GiB from 72) parks switched-away chats —
  an 111K-token chat comes back in ~1 s.
- `--ple-io ram` at 96 GB+ — mlock the 28.8 GB n-gram table instead of reading it over SSD;
  a cold 32K prompt used to wait ~3 s on those reads.
- the official sampling block (temperature 1.0, top_p 0.95, top_k 20). Greedy Qwen3.8 has
  documented reasoning loops, so if you strip it, you own the consequences.

[`configs/`](configs/) holds the exact configs from our machine (our absolute paths inside —
reference, not templates).

## Gotchas we hit so you don't

- **One request at a time**, by design here. A second queues.
- No other GPU engine at the same time (vLLM, another Strata): both cards and 60-110 GB of
  RAM are in use while this runs.
- A desktop on one card costs the peer tier a few hundred experts and can reorder cards:
  `GPUS_ORDER=1,0` and keep the monitor off the first card.
- The server refuses Host names it was not told to answer (DNS-rebinding protection): no
  API key + LAN access = set `ALLOWED_HOSTS`.
- The 3090 pair is loud and hot under prefill; the peer path idles the helper between
  prompts, so watch thermals on first runs.

## Not in this build

Two-stream "elastic" operation (the second card becoming a second engine on a second
request — our later work), the layer-split mode (upstream's [docs/MULTI_GPU.md](../docs/MULTI_GPU.md)
covers it), the Coder/Swift variants, and Windows containers. The engine itself still runs
single-card exactly as upstream.

## Provenance

Fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT, copyright its authors),
based on tag `v0.1.38`: `git log v0.1.38..main --oneline` lists the 29 commits — our
peer tier and prefill fast paths, plus the upstream PR branches #603, #652 and #583 merged
in (each A/B'd and gate-tested here first). Parts of this work were merged upstream along
the way: #186-#188, #202 (`--ple-io ram`), #203, #477 and #531 (`--peer-device`); the
exactness switches our gate uses are #152. Bench tools: [`verify/`](verify/).

Everything in `dual3090/` is ours under the same MIT terms. Questions and measurements
welcome as issues.
