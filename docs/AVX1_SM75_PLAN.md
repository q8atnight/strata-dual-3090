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
