#!/usr/bin/env python3
"""prefill_probe - how far a change moves greedy answers after long prompts, against a reference run.

Without logprobs from the server, a rounding change in prompt reading is judged by its effect on what follows:
`n` long code prompts (different slices of corpus.txt, so nothing is reused), greedy, `gen` answer tokens each.
`compare` reports, against a reference run: first-token agreement, fully identical answers, and the mean position of
the first different character - to be read next to a rerun of the reference (the noise floor) and a change Strata
already accepts (another prompt chunk size).

  python3 prefill_probe.py run --url http://127.0.0.1:8080/v1 --out A.json
  python3 prefill_probe.py compare A.json B.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request

from flashbench import corpus

QUESTION = "\n</source>\n\nContinue the source above: write the next 40 lines of code."


def ask(url, text, gen, timeout):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": gen, "temperature": 0,
            "seed": 1234, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"]


def run(a):
    text = corpus()
    out = []
    for i in range(a.n):
        chars = int((8000 + (16000 * i) // max(1, a.n - 1)) * 3.3)        # 8K .. 24K tokens
        start = (i * 7919 * 97) % max(1, len(text) - chars)               # a different slice each time
        prompt = f"[probe {i}]\n<source>\n" + text[start:start + chars] + QUESTION
        ans, ntok = ask(a.url, prompt, a.gen, a.timeout)
        out.append({"i": i, "prompt_tokens": ntok, "text": ans})
        print(f"{i:3d}: {ntok:6d} tok  {ans[:60]!r}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    first = same = 0
    pos = []
    for p, q in zip(x, y):
        s, t = p["text"], q["text"]
        k = next((i for i, (c, d) in enumerate(zip(s, t)) if c != d), min(len(s), len(t)))
        if s == t:
            same += 1
            k = len(s)
        first += s[:1] == t[:1] and (k > 0 or s == t)
        pos.append(k)
    n = len(pos)
    print(f"{a.b} vs {a.a}: {n} prompts, identical answers {same}/{n}, same start {first}/{n}, "
          f"first difference at char {statistics.mean(pos):.0f} on average (median {statistics.median(pos):.0f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=24)
    r.add_argument("--gen", type=int, default=48)
    r.add_argument("--timeout", type=int, default=3600)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    {"run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
