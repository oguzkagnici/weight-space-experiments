"""Feasibility probe for thinking-mode evaluation on CPU: throughput, thinking lengths, cap hits, accuracy."""
import sys, time, re, json, random, torch
sys.path.insert(0, "/home/user/weight-space-experiments")
from transformers import AutoTokenizer, AutoModelForCausalLM
from wsp import data
torch.set_num_threads(4)
MD = "/home/user/models/LFM3-1.7B"
bs = int(sys.argv[1]); cap = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else bs
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, trust_remote_code=True).eval()
Q = data.questions(); ids = random.Random(5).sample(data.splits()["practice"], n)
INSTR = 'Think step by step, then finish with "The answer is (X)" where X is the letter of the correct option.'
def prompt(q):
    opts = "\n".join(f"({'ABCDEFGHIJ'[j]}) {o}" for j, o in enumerate(q["options"]))
    return tok.apply_chat_template([{"role": "user", "content": f"{q['question']}\n\n{opts}\n\n{INSTR}"}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=True)
def extract(text):
    ans = text.split("</think>")[-1] if "</think>" in text else ""
    m = re.findall(r"answer is \(?([A-J])\)?", ans)
    return m[-1] if m else None
res = []; t0 = time.time(); gen_tok = 0
for b in range(0, n, bs):
    batch = ids[b:b + bs]
    enc = tok([prompt(Q[i]) for i in batch], return_tensors="pt", padding=True, add_special_tokens=False)
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    dt = time.time() - t
    for i, qid in enumerate(batch):
        g = out[i, enc.input_ids.shape[1]:]
        L = int((g != tok.pad_token_id).sum()); gen_tok += L
        text = tok.decode(g, skip_special_tokens=False)
        closed = "</think>" in text; p = extract(text)
        res.append({"id": qid, "len": L, "closed": closed, "pred": p, "gold": Q[qid]["answer"], "correct": p == Q[qid]["answer"]})
    print(f"batch {b//bs}: {dt:.0f}s, {sum(r['len'] for r in res[-len(batch):])/dt:.1f} tok/s", flush=True)
L = sorted(r["len"] for r in res)
print(f"n={len(res)} acc={sum(r['correct'] for r in res)/len(res):.3f} closed_think={sum(r['closed'] for r in res)/len(res):.2f} "
      f"no_answer={sum(r['pred'] is None for r in res)} len median={L[len(L)//2]} max={L[-1]} hit_cap={sum(l>=cap for l in L)} "
      f"total {time.time()-t0:.0f}s, overall {gen_tok/(time.time()-t0):.1f} gen tok/s")
json.dump(res, open(f"/tmp/claude-0/-home-user-weight-space-experiments/dfc1b52a-462b-5676-a560-7a430e6a0613/scratchpad/think_probe_{bs}_{cap}.json", "w"))
