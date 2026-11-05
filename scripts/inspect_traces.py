"""Regenerate a few truncated questions and measure repetition in the full reasoning trace."""
import sys, json, re, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/user/weight-space-experiments")
from wsp import data
port, cap = 8091, 3072
r = json.load(open(sys.argv[1])); ids = [x["id"] for x in r if x["finish"] == "length"][:4]
Q = data.questions()
INSTR = 'Think step by step, then finish with "The answer is (X)" where X is the letter of the correct option.'
def ask(qid):
    q = Q[qid]; opts = "\n".join(f"({'ABCDEFGHIJ'[j]}) {o}" for j, o in enumerate(q["options"]))
    body = {"messages": [{"role": "user", "content": f"{q['question']}\n\n{opts}\n\n{INSTR}"}], "max_tokens": cap,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": hash(qid) % 100000}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    c = json.load(urllib.request.urlopen(req, timeout=36000))["choices"][0]["message"]
    return qid, (c.get("reasoning_content") or "") + (c.get("content") or "")
with cf.ThreadPoolExecutor(4) as ex:
    out = list(ex.map(ask, ids))
for qid, t in out:
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    uniq = len(set(lines)) / max(1, len(lines))
    # share of 40-char windows (step 20) that occurred earlier in the text
    w = [t[i:i + 40] for i in range(0, max(0, len(t) - 40), 20)]; seen = set(); rep = 0
    for x in w:
        rep += x in seen; seen.add(x)
    asc = lambda s: s.encode("ascii", "replace").decode()
    print(f"\n=== {qid}: {len(t)} chars, {len(lines)} lines, unique-line share {uniq:.2f}, repeated 40-char windows {rep/max(1,len(w)):.2f}")
    print("  [middle]", asc(t[len(t)//2: len(t)//2 + 300]).replace("\n", " | "))
    print("  [end]   ", asc(t[-300:]).replace("\n", " | "))
json.dump(dict(out), open("/tmp/claude-0/-home-user-weight-space-experiments/dfc1b52a-462b-5676-a560-7a430e6a0613/scratchpad/traces.json", "w"))
