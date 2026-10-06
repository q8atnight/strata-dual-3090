# flashbench — speed test for any local AI engine

One small program that measures how fast an AI server **reads a prompt** (prefill) and **writes an
answer** (decode). It sends exactly the same requests to every engine, so results from vLLM, Strata,
llama.cpp — today or in six months — can be put side by side.

It needs only Python 3 (nothing to install) and works against any server that speaks the OpenAI API.

## Where it lives
This folder (`dual3090/verify/`) in the strata-dual-3090 repo.

Files: `flashbench.py` (the program), `corpus.txt` (the fixed test text: public llama.cpp source code,
MIT licence), `cuts.json` (how much of the text makes 1K / 8K / … tokens), `results/` (every run, one
JSON file each), this README.

## The three commands you need

Run them in a terminal on the PC, inside the bench folder (`cd ~/linux2/strata-work/bench`).

**1. Full test** (about 15–40 minutes, depending on the engine):
```
python3 flashbench.py run --url http://127.0.0.1:8000/v1 --label vllm-today
```
- `--url` is the engine's address: vLLM = port **8000**, Strata = port **8080**.
- `--label` is any short name you choose; it becomes part of the result file name.
- Add `--note "what was special"` to remember settings (e.g. `--note "hot cache 80"`).

**2. Quick check** (2–5 minutes: only 1K and 32K prompts, one run each):
```
python3 flashbench.py run --url http://127.0.0.1:8080/v1 --label strata-quick --quick
```

**3. Compare results** (table, and a chart picture):
```
python3 flashbench.py compare results/*.json --png compare.png
```
Pick specific files instead of `results/*.json` to compare only those.

## What it measures

| | what | how |
|---|---|---|
| **Prefill** (prompt reading) | how fast a long prompt is taken in | prompt tokens ÷ time until the first answer token |
| **Decode** (writing) | how fast the answer appears | answer tokens ÷ time from first to last token |

- **Prompt lengths:** 1K, 8K, 32K, 64K, 128K tokens (change with `--contexts 1k,32k,200k`).
- **Two ways of writing:**
  - `greedy` — the model always takes its top choice (temperature 0): best case, and the number
    engine authors usually publish.
  - `sampled` — temperature 0.7 (top_p 0.95, top_k 20): like everyday use. Usually a little slower,
    because the engines' "guess ahead" trick lands less often.
- **Repeats:** 3 runs per length and mode (2 at 128K and above); the table shows the median.
- **Answer length:** 512 tokens, thinking switched off (`--gen 1024` for longer).
- **Warm-up:** one short request first, not counted.
- **Fair play:** every request starts with a new random id, so no engine can re-use a prompt it has
  already read and look faster than it is.

## Reading the results
- Prefill at **1K** is always low: at that size the fixed start-up cost of a request dominates. Judge
  prefill at 8K and above.
- A line marked `(SHORT answer)` means the model stopped before half the requested length — its
  decode number is less reliable.
- Only compare numbers from the same machine; the engine and its settings are what you're testing.

## Advanced
- **Exact-output check** (used when changing an engine — does it still write *exactly* the same text?):
  ```
  python3 flashbench.py gate --url http://127.0.0.1:8080/v1 --save ref-stock.json     # once, on the original
  python3 flashbench.py gate --url http://127.0.0.1:8080/v1 --against ref-stock.json  # after a change
  ```
  Prints `GATE: PASS` or where the text first differs.
- `--api-key KEY` if the server needs a key; `--modes greedy` to skip the sampled runs;
  `--repeats 5` for more precision; `--keep-text` stores the generated answers too.
- `calibrate` (maintainers only) rebuilds `cuts.json` — only needed if `corpus.txt` or the prompt
  wording in `flashbench.py` is changed, which would make old and new results incomparable. Don't.

## History
- 1.0 (2026-09-28): created for the Strata-on-2×3090 project; baseline = vLLM Flash-Next live setup.
