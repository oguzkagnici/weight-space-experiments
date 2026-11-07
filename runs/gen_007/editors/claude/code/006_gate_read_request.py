# Hypothesis: the "end of a two-finalist paragraph" state is linearly readable at the L22/L23 MLP inputs, as at L21 in gen 6.
# Builds the read request: 32 DEV bank traces, feed_forward inputs at L22 and L23, positions all.
import json, random
from wsp.editor import *
rows = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl")}
g = json.load(open("work/groups.json"))
rng = random.Random(70)
others = rng.sample(g["others"], 10)
dev = g["narrow"] + g["fin_narrow"] + others
texts = [trace_text(rows[i]) for i in dev]
n = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
print(len(dev), "texts", sum(n), "tokens, max", max(n), "others:", [(i, rows[i]["cut"]) for i in others])
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"inputs": ["model.layers.22.feed_forward", "model.layers.23.feed_forward"], "positions": "all", "token_logprobs": True}},
          open("work/req_02.json", "w"))
json.dump({"dev": dev, "others": others}, open("work/dev_ids.json", "w"))
