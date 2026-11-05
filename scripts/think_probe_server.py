"""Thinking-length probe against a running llama-server (OpenAI-compatible API), continuous batching."""
import sys, json, re, time, random, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/user/weight-space-experiments")
from wsp import data
port, n, cap, par = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Q = data.questions(); ids = random.Random(5).sample(data.splits()["practice"], n)
INSTR = 'Think step by step, then finish with "The answer is (X)" where X is the letter of the correct option.'
def ask(qid):
    q = Q[qid]; opts = "\n".join(f"({'ABCDEFGHIJ'[j]}) {o}" for j, o in enumerate(q["options"]))
    body = {"messages": [{"role": "user", "content": f"{q['question']}\n\n{opts}\n\n{INSTR}"}], "max_tokens": cap,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": hash(qid) % 100000}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=36000))
    c = r["choices"][0]; text = (c["message"].get("reasoning_content") or "") + "\n</think>\n" + (c["message"].get("content") or "")
    after = c["message"].get("content") or ""
    m = re.findall(r"answer is \(?([A-J])\)?", after)
    return {"id": qid, "tokens": r["usage"]["completion_tokens"], "finish": c["finish_reason"],
            "think_chars": len(c["message"].get("reasoning_content") or ""), "pred": m[-1] if m else None,
            "gold": q["answer"], "correct": bool(m) and m[-1] == q["answer"], "answer_tail": after[-120:]}
t0 = time.time()
with cf.ThreadPoolExecutor(par) as ex:
    res = list(ex.map(ask, ids))
dt = time.time() - t0
L = sorted(r["tokens"] for r in res); tot = sum(L)
print(f"n={n} acc={sum(r['correct'] for r in res)/n:.3f} no_answer={sum(r['pred'] is None for r in res)} "
      f"hit_cap={sum(r['finish']=='length' for r in res)} tokens median={L[n//2]} p75={L[3*n//4]} max={L[-1]} mean={tot/n:.0f} "
      f"| {dt:.0f}s total, {tot/dt:.1f} gen tok/s")
json.dump(res, open(f"/tmp/claude-0/-home-user-weight-space-experiments/dfc1b52a-462b-5676-a560-7a430e6a0613/scratchpad/server_probe_{n}_{cap}.json", "w"), indent=1)
