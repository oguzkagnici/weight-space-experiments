# Hypothesis (H1): some MLP neurons have |h_j| below 1/254 of their 32-neuron block's absmax on (almost) every token,
# so their contribution is exactly 0 in llama.cpp's Q8_0 down projection. This builds the screening read request.
import json, random
from wsp.editor import *
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl")]
cut = [r for r in rows if r["cut"]]; fin = [r for r in rows if not r["cut"]]
rng = random.Random(7)
pick = rng.sample(cut, 2) + rng.sample(fin, 2)
texts = []
for r in pick:
    t = trace_text(r)
    p = render_think(r["id"])
    gen_ids = tok.encode(t[len(p):], add_special_tokens=False)[:1200]
    texts.append(p + tok.decode(gen_ids))
print([(r["id"], r["category"], r["cut"]) for r in pick], [len(tok.encode(t, add_special_tokens=False)) for t in texts])
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"mlp": list(range(12, 24)), "positions": "all"}}, open("work/req_01.json", "w"))
json.dump([r["id"] for r in pick], open("work/screen_ids.json", "w"))
