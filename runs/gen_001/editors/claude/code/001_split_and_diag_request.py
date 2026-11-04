# hypothesis: none yet - set up a DEV/HOLDOUT split of practice and a diagnostic request (512 DEV questions, stratified)
import json, random, collections
from wsp.editor import *
Q = questions(); ids = sorted(Q)
rng = random.Random(1234); rng.shuffle(ids)
holdout, dev = ids[:1000], ids[1000:]
json.dump({"dev": dev, "holdout": holdout}, open("work/split.json", "w"))
by = collections.defaultdict(list)
for q in dev: by[Q[q]["category"]].append(q)
diag = []
for c in sorted(by): diag += by[c][:37]
diag = diag[:512]
attn_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
req = {"type": "score", "model": "original", "questions": diag,
       "activations": {"hidden_states": [8, 12, 16, 18, 20, 22, 24],
                       "inputs": [f"model.layers.{l}.self_attn.o_proj" for l in attn_layers]}}
json.dump(req, open("work/req_01_diag.json", "w"))
print(len(dev), len(holdout), len(diag), collections.Counter(Q[q]["category"] for q in diag).most_common(3))
