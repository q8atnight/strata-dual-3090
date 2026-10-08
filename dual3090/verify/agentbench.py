#!/usr/bin/env python3
"""agentbench - a coding-agent session against an OpenAI-compatible server: one conversation that starts with a
large code context and grows by one "next file" per turn, like an agent reading a repository.

Per turn it records the new prompt tokens, the time to the first answer token, decode speed and (with
--engine-log) Strata's own "prompt N tokens = R reused + K read" line.  Greedy, thinking off.

  python3 agentbench.py run --url http://127.0.0.1:8080/v1 --label base --engine-log ~/strata-dual-3090/strata-iq3_xxs.log
  python3 agentbench.py run ... --fixed-id --keep-text      (exactness: same prompts every run; start a fresh server)
  python3 agentbench.py check a.json b.json                 (is every turn's text identical?)
  python3 agentbench.py compare a.json b.json               (b against a, turn by turn: engine read time, decode)
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import re
import secrets
import statistics
import time
import urllib.parse

from flashbench import HEAD, corpus, headers, parse_ctx, server_model

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS_PER_TOKEN = 3.3   # the corpus, measured by flashbench's cuts.json
FIRST_Q = ("\n</source>\n\nReview the source above: what it does, and its three most serious bugs or risks, "
           "each with a short fix.")
NEXT_Q = "\n</source>\n\nContinue the review with this next part: what changed, and one bug or risk with a fix."
ENGINE_LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")


def stream_messages(base, model, messages, max_tokens, timeout):
    u = urllib.parse.urlparse(base.rstrip("/") + "/chat/completions")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": False}}
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", u.path, json.dumps(body), headers())
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    t_first = t_last = None
    usage, out = None, []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece:
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
                out.append(piece)
    conn.close()
    gen = (usage or {}).get("completion_tokens", 0)
    ttft = (t_first - t0) if t_first else float("nan")
    dec = (gen - 1) / (t_last - t_first) if t_first and t_last > t_first and gen > 1 else float("nan")
    return {"prompt_tokens": (usage or {}).get("prompt_tokens", 0), "gen_tokens": gen, "ttft_s": ttft,
            "decode_tps": dec, "text": "".join(out)}


def engine_line(path, after):
    """The last 'prompt ... read in' line the engine log got past byte offset `after`."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(after)
        hits = ENGINE_LINE.findall(f.read().decode("utf-8", "replace"))
    if not hits:
        return None
    total, reused, read, ms = map(int, hits[-1])
    return {"total": total, "reused": reused, "read": read, "ms": ms}


def run(a):
    model = a.model or server_model(a.url)[0]
    text = corpus()
    start, step, until = parse_ctx(a.start), parse_ctx(a.step), parse_ctx(a.until)
    mark = "fixed" if a.fixed_id else secrets.token_hex(8)
    pos = int(start * CHARS_PER_TOKEN)
    messages = [{"role": "user", "content": f"[agent-bench {mark}]\n" + HEAD + text[:pos] + FIRST_Q}]
    turns = []
    log_path = os.path.expanduser(a.engine_log) if a.engine_log else None
    while True:
        off = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0
        r = stream_messages(a.url, model, messages, a.gen, a.timeout)
        r["engine"] = engine_line(log_path, off)
        r["turn"] = len(turns)
        if not a.keep_text:
            reply = r.pop("text")
        else:
            reply = r["text"]
        turns.append(r)
        e = r["engine"]
        print(f"turn {r['turn']:2d}: prompt {r['prompt_tokens']:6d} tok  ttft {r['ttft_s']:6.2f} s  "
              + (f"engine {e['reused']:6d} reused + {e['read']:5d} read in {e['ms']:6d} ms "
                 f"({1000 * e['read'] / max(e['ms'], 1):6.1f} tok/s)  " if e else "")
              + f"decode {r['decode_tps']:5.1f} tok/s", flush=True)
        if r["prompt_tokens"] + step + a.gen > until or pos >= len(text):
            break
        nxt = int(step * CHARS_PER_TOKEN)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": HEAD + text[pos:pos + nxt] + NEXT_Q})
        pos += nxt
    res = {"label": a.label, "note": a.note, "model": model, "date": dt.datetime.now().isoformat(timespec="seconds"),
           "start": start, "step": step, "until": until, "gen": a.gen, "fixed_id": a.fixed_id, "turns": turns}
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_agent_{a.label}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    inc = [t for t in turns[1:] if t["engine"]]
    if inc:
        print(f"median incremental read: {statistics.median(1000 * t['engine']['read'] / max(t['engine']['ms'], 1) for t in inc):.1f} tok/s"
              f"; median decode {statistics.median(t['decode_tps'] for t in turns):.1f} tok/s")
    print("saved:", path)


def check(a):
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    ok = True
    for t, u in zip(x["turns"], y["turns"]):
        if "text" not in t or "text" not in u:
            raise SystemExit("both runs need --keep-text")
        same = t["text"] == u["text"]
        ok &= same
        if not same:
            i = next((i for i, (p, q) in enumerate(zip(t["text"], u["text"])) if p != q), min(len(t["text"]), len(u["text"])))
            print(f"turn {t['turn']} (prompt {t['prompt_tokens']} tok): DIFFERS at char {i}")
        else:
            print(f"turn {t['turn']} (prompt {t['prompt_tokens']} tok): IDENTICAL")
    if len(x["turns"]) != len(y["turns"]):
        ok = False
        print("different number of turns")
    print("AGENT GATE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def compare(a):
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    reads, decs = [], []
    print(f"{'turn':>4} {'prompt':>7} {'read A ms':>10} {'read B ms':>10} {'B/A':>6} {'dec A':>6} {'dec B':>6} {'B/A':>6}")
    for t, u in zip(x["turns"], y["turns"]):
        same = t.get("text") is not None and t.get("text") == u.get("text")
        e, f = t.get("engine"), u.get("engine")
        r = f["ms"] / e["ms"] if e and f and e["ms"] else float("nan")
        d = u["decode_tps"] / t["decode_tps"] if t["decode_tps"] else float("nan")
        if t["turn"] > 0 and e and f:
            reads.append(r)
        if same:
            decs.append(d)
        print(f"{t['turn']:4d} {t['prompt_tokens']:7d} {e['ms'] if e else 0:10d} {f['ms'] if f else 0:10d} {r:6.3f} "
              f"{t['decode_tps']:6.1f} {u['decode_tps']:6.1f} {d:6.3f}{'' if same else '  (text differs)'}")
    if reads:
        print(f"median B/A: incremental read time {statistics.median(reads):.3f}", end="")
    if decs:
        print(f", decode speed {statistics.median(decs):.3f} (over {len(decs)} turns with identical text)", end="")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--note", default="")
    r.add_argument("--model")
    r.add_argument("--start", default="32k", help="first prompt size (default 32k)")
    r.add_argument("--step", default="4k", help="tokens added per turn (default 4k)")
    r.add_argument("--until", default="192k", help="stop before the prompt passes this (default 192k)")
    r.add_argument("--gen", type=int, default=128, help="answer tokens per turn (default 128)")
    r.add_argument("--fixed-id", action="store_true", help="the same prompts every run (for exactness checks)")
    r.add_argument("--keep-text", action="store_true")
    r.add_argument("--engine-log", help="Strata's engine log, for its reused/read numbers")
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--outdir", default=os.path.join(HERE, "results"))
    c = sub.add_parser("check")
    c.add_argument("a")
    c.add_argument("b")
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    a = p.parse_args()
    {"run": run, "check": check, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
