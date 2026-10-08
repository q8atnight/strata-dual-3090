# Plan: Strata on 2× Xeon E5-2670 v2 (AVX1) + 2× CMP 50HX (SM75), IQ3_XXS, coding use

Status (2026-10-08): two exact changes measured and kept; the rest is ranked by profile. Use case: coding agent,
contexts to ~200K, a few K new tokens per turn on a reused prefix. Rule: **bit-exact only** (same text word for word).

## 1. The machine (read from `xrip@192.168.1.224`)

| part | fact |
|---|---|
| CPU | 2× Xeon E5-2670 v2 (Ivy Bridge-EP), 2 × 10 cores, 40 threads; AVX, F16C, SSSE3/SSE4.x; no AVX2/FMA/BMI2 |
| NUMA | 2 nodes × 32 GB; both GPUs on node 0 |
| GPU | 2× CMP 50HX (TU102, sm_75, 56 SMs), unlocked to RTX 2080 speed, 20 GB each |
| Link | PCIe Gen2 x16, engine probe 6.2 / 6.1 GB/s host→device; P2P on (PHB path) |
| Build | sm_75, `GGML_NATIVE=ON` built on the box (AVX + F16C ggml) |
| Model | IQ3_XXS pack: down Q2_0 in 30 layers, IQ4_NL in 18; `--layer-split auto` |

## 2. Tools made for this

- `dual3090/verify/agentbench.py` — a coding-agent session: 32K of C source, then +~4.4K tokens per turn to ~190K,
  greedy, 128 answer tokens. Records time to first token, the engine's "reused + read" numbers, decode. With
  `--fixed-id --keep-text` and a fresh server per run, `agentbench.py check A B` is the exactness gate.
- `src/kernels/qsa_scores_parity.cpp` — GPU test: QSA block scores, new kernel vs old, bitwise, plus times.
- Note: running the same long prompt twice on one server is **not** an exactness test: the second run reuses the
  conversation cache, which rounds differently. Each arm needs a fresh server.

## 3. Results (agent session 34K→190K, same prompts, fresh server each, 262144 context)

| | base | now | change |
|---|---:|---:|---:|
| time to first token at 190K | 10.49 s | 7.13 s | −32 % |
| incremental read at 190K | 476 tok/s | 632 tok/s | +33 % |
| median incremental read (all turns) | 576 tok/s | 682 tok/s | +18 % |
| median decode | 73.3 tok/s | 73.2 tok/s | same |
| text of all 36 turns | — | identical | AGENT GATE: PASS |

Kept changes:
1. **Tokenizer cache** (`tools/strata_tokenizer.py`): the pure-Python BPE re-encoded the whole conversation every
   turn: 1.06 s at 190K tokens on this CPU. Text between special tokens is encoded on its own, so long pieces are
   cached: 0.042 s. Ids identical on all 36 turns; `serve.test_detok` passes with the real tokenizer.
2. **QSA block scores on Turing** (`src/kernels/cuda/qsa_select.cu`, `block_scores_thread_kernel`): the prompt path
   used a warp per (query, block) with 20 shuffles per score. Now a lane per (query, head), keys broadcast from
   shared memory, and the 32 partials added in the same tree as the shuffles. 141 M scores bitwise identical on both
   cards; 4.6-5.2x faster per call (1,024 queries at 200K: 52.4 → 10.8 ms). `STRATA_SCORES_THREAD=0` = old kernel.

3. **AVX1 Q2_0 rows** (`src/kernels/cpu/avx1.cpp`, chosen when the CPU has AVX but not AVX2): bitwise equal to the
   scalar reference (`cpu_avx1_parity`: all 65,536 half scales, 216 random cases); one expert's down rows on one
   E5-2670 v2 core: 3.86 → 0.41 ms (1 token, 9.4x), 13.2 → 0.78 ms (4 tokens, 16.9x).
4. **Router lookahead without AVX2**: `bf16_rows_dot_multi_avx1` (multiply + add), so the file-tier prefetch no
   longer runs AVX2/FMA code on this CPU (it was a SIGILL path; off in the current RAM-arena config).
5. **GCC 14.2** for C, C++ and the CUDA host (`build-gcc14`), same options as `build`.
6. **Gate/up MMQ reads the experts in place** (patched copy of llama.cpp's `mmq.cuh` in the build tree): copy kernel
   75.5 → 31.9 ms per card per turn, engine read −0.7 %. Exact (agent gate PASS). `STRATA_MMQ_INPLACE=0` = old path.
7. **Grid codebooks in shared memory on Turing** (decode expert kernels; from xrip/llama.cpp-avx1-numa-sm75
   1536af3b9): `iq_multi_parity` 0 failures; decode +2.0 %.
8. **`STRATA_HC_FP16=1`** (opt-in, **not bit-exact**): hc read's BF16 GEMMs on FP16 tensor cores. Incremental read
   −7.7 % per turn, cold 34K −13.6 %. Quality (`prefill_probe.py`, 24 prompts 8K-24K): 20/24 answers identical,
   24/24 same first token; for scale, an exact rerun 24/24 / 24/24 and `--prefill 4096` 17/24 / 23/24.
9. **Prompt attention, int8 K/V on Turing**: next chunk prefetched in registers, K and V sharing one shared buffer
   (37.9 → 29.2 KB, two blocks per SM). Bitwise identical (output checksums); kernel −40 % (2,048 queries at 140K:
   21.94 → 13.31 ms); incremental read −3.5 % per turn; agent gate PASS.
10. **Decode hc read, up projection** (`gr_up_multi_kernel`): `lo` staged as two planes (every read had hit its
    shared-memory bank twice: 415K of 826K wavefronts were conflicts) and two rows per pass with the (row, token)
    reductions interleaved (ncu: stalls were `short_scoreboard`). Bitwise identical (`hc_read_bench` checksums for
    1-8 tokens, `gr_parity`); kernel 30.0 → 23.6 µs at 4 tokens; one hc read call −6 % to −11 % at 3-8 tokens
    (+2 % at 1); agent session: identical text on all 36 turns, decode median +1.5 %.
    Measured and dropped for the down projection: 54 blocks of 6 warps instead of 41 of 8 (+9 %), weights two tiles
    ahead (slower too).
11. **MMQ: the IQ3_XXS codebook in shared memory** (configure-time patch of the build tree's `mmq.cuh` /
    `mmq-load-tiles.cuh` copies, each text required exactly once): the tile loader's 8 scattered 4-byte lookups per
    thread and tile now hit a 1 KB table that `mul_mat_q<IQ3_XXS>` copies to shared memory first. Same checksums
    (`prefill_mmq_bench`, 87 and 400 rows per expert); gate/up −3.5..−3.9 %, down (Q2_0) unchanged; agent session back
    to back with the base: identical text on all 36 turns, incremental read median −0.3 % (CUDA1 is PCIe-bound, see
    below).

Measured and dropped (bitwise identical, not faster):
- **Dense decode GEMVs with the columns in shared memory** (`native_mmvq_multi_kernel`; ncu: `lg_throttle` first):
  blocks that copy the Q8_1 columns once and loop over rows. Head −6 %, IQ4_XS −8..−21 % at 2-4 columns, but Q6_K
  +8..+25 % and most shapes much slower at 6-8 columns (fewer resident warps); with four row groups per block still
  mixed (`mmvq_bench`).
- **MMQ tile-load pipelining**: ncu: `mul_mat_q` holds 255 registers and 53 KB of shared memory, one 8-warp block
  per SM, issue slots 25-29 % busy, no single stall; pipelining its tile loads needs registers it does not have, and
  two blocks per SM (I = 64) gave nothing earlier. Not attempted beyond the codebook step (item 11).

**One incremental turn at ~140K (+4.4K tokens), nsys, current binary**: the cards read the prompt one after the
other (CUDA0 2.77 s, then CUDA1 2.58 s). CUDA0 is kernel-bound (kernels 2.58 s; 11 GB of expert uploads take 1.87 s
beside them); CUDA1 is PCIe-bound (14 GB of uploads take 2.45 s of its 2.58 s; kernels 2.2 s), so faster kernels
help CUDA1 little. Kernels per card: `mul_mat_q` 821 / 690 ms, magma BF16 (hc read) 684 / 540, `prompt_attn` 181 /
181, `block_scores` 145 / 139, FP16 GEMMs 184 / 153, GDN 118 / 94. Overlapping the two cards needs more than one
chunk per turn, and the chunk size changes the text: not an exact step.

Measured and dropped: `--kv fp16` (more precise than int8, attention 3.5x faster) — but twice the KV bytes to stage
and read: incremental read **+7.6 %** per turn.

**Decode is not always reproducible run to run** on this box: one of three runs of the same binary diverged at turn
8 of the agent session. A single gate failure needs a rerun; the kernel parity tests are the strict proof. Suspect:
the adaptive expert swaps (`--adapt-swaps`, on by default) on the second card of the layer split.

Measured and not kept (within ~5 %, and chunk size changes the text): `--prefill 4096/2048`,
`STRATA_SPLIT_SMALL_OWN=2304/4608`, `--kv-resident 262144`.

**Peer mode loses on this link** (agent session 103K→150K, new engine): layer split vs peer (QSA split 0.5 + GDN
split, `mkconfig.py` defaults): cold 103K read 1,692 vs 685 tok/s, incremental read 684 vs 540 tok/s, decode
75.4 vs 62.7 tok/s. The peer tier was built for NVLink; over PCIe Gen2 P2P its expert traffic costs more than the
split gains. Layer split stays.

## 4. Where the time goes now (nsys, one +4.4K turn at 140K, per card; before the scorer change)

| kernel | ms | note |
|---|---:|---|
| `block_scores_kernel` | 1,045 | fixed above (now ~200) |
| `mul_mat_q` | 716 | ggml MMQ experts, prompt |
| `magma_sgemmEx_kernel` | 678 | cuBLAS BF16 GEMM without tensor cores (hc read); exact change not possible |
| `wait_flag_ge_kernel` | 520 | decode: GPU spins on a flag (CPU misses or the other card) |
| `prompt_attn_kernel` | 325 | QSA attention, prompt |

And structure: in layer split a turn's read is one chunk, so CUDA1 waits for CUDA0 (~4 s each, one after another).

Decode (nsys with graph nodes, 32K context, 512-token answer; ~214 windows of ~46 ms; per card and window):
dense quantized GEMVs (`native_mmvq_multi_kernel`) 3.1 / 4.2 ms, expert kernels 4.5 / 4.1 ms, hc read 2.6 / 2.5 ms,
BF16 GEMVs 0.7 ms; the rest is the other card's turn (`wait_flag_ge_kernel`). Peak read bandwidth of a card: 589 GB/s
(`__ldg` of 16 bytes). The dense GEMVs reach 450-560 GB/s for one column but 180-310 GB/s for four (`mmvq_bench`).

**Timing tests and the CMP idle governor**: `cmp-idle-governor.service` forces P8 on an idle card and restores the
load profile only after a few seconds of work, so short benches start at 645 MHz. `~/opt-avx1/perf.sh <cmd>` stops
the governor and the CPU's C-states deeper than C1 for the test and always puts both back (idle cards run hot without
the governor).

**`pcie_frac` from the startup probe changes the text**: CUDA1's PCIe probe gave 6.0 GB/s in one run (pcie_frac 0.16)
and 6.1 GB/s in others (0.17); the share of misses sent over PCIe moves experts between GPU and CPU, which round
differently, and the text changed at turn 33 of 36. Gates now pass `--pcie-frac 0.17`, and the base and the new
binary run back to back in one job.

## 5. Next steps (ranked)

| # | step | type | exact? | gate |
|---|---|---|---|---|
| 1 | ~~Peer mode + QSA/GDN split~~ — measured, loses (section 3) | config | — | done |
| 2 | ~~`wait_flag_ge_kernel` in decode~~ — `STRATA_VERIFY_PROFILE`: waitA 0.7 ms, waitCPU 0.1 ms per window; the CPU is not on the decode path. Decode is GPU work across many small kernels (expert kernels 31 %) | measure | — | done |
| 3 | ~~AVX1 Q2_0 rows~~ — done (section 3); the activation quantizer is still scalar (small) | code | yes | done |
| 4 | ~~Router lookahead AVX1~~ — done (section 3) | code | n/a | done |
| 5 | `--pcie-frac`: "auto" parses as 0 (`atof`); dual `mkconfig.py` forces 0; `STRATA_PEER_HOT_AT=8700` is a 3090 value | config/code | — | agent session |
| 6 | ~~hc read~~ — `STRATA_HC_FP16=1` (section 3); MMQ (memory latency at 8 warps/SM, needs pipelining in ggml), prompt attention (300 ms per card) | code | MMQ/attn: yes | — |
| 7 | Decode nondeterminism: repeat the agent session with `--adapt-swaps 0` | measure | — | 3 identical runs |

Ideas taken from `Strata для AVX1 SM75 и IQ3_XXS.md`: peer prefill + QSA/GDN split A/B, the lookahead SIGILL path,
`--pcie-frac` parsing, the Q2_0 kernel contract, the weak Q2 check in `native_expert_parity`. Its larger part (one
fixed Ivy Bridge build profile, removing the AVX2/AVX-512 sources) is a refactor, not a speed step; open question.

## 6. How a change is verified
1. Kernel level: a parity test, bitwise against the path it replaces, on both cards.
2. End to end: `agentbench.py run --fixed-id --keep-text` on a fresh server, `agentbench.py check` against the base
   run: every turn identical.
3. Speed: the same agent session; compare per-turn engine read times at the same depth and time to first token.
