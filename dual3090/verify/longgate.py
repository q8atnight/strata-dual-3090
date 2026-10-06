#!/usr/bin/env python3
# longgate.py save|check <file> [tokens]: a fixed long prompt (corpus), greedy 200 tokens, compare the text exactly
import json, sys, urllib.request, os
BASE = os.environ.get("FB_URL", "http://127.0.0.1:8080/v1")
mode, f = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
H = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(H, "corpus.txt")).read()[: n * 3]
out = {}
for name, q in [("sum", "Summarize what this code does in detail."), ("fn", "List the ten most important functions and what each does.")]:
    body = {"model": "x", "messages": [{"role": "user", "content": text + "\n\n" + q}], "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    out[name] = r["choices"][0]["message"]["content"]
    print(name, r["usage"]["prompt_tokens"], "tok")
if mode == "save":
    json.dump(out, open(f, "w")); print("saved", f)
else:
    ref = json.load(open(f)); ok = True
    for k in ref:
        same = ref[k] == out[k]; ok &= same
        if not same:
            i = next((i for i, (a, b) in enumerate(zip(ref[k], out[k])) if a != b), min(len(ref[k]), len(out[k])))
            print(k, "DIFFERS at char", i)
        else: print(k, "IDENTICAL")
    print("LONG GATE:", "PASS" if ok else "FAIL")
