# Pass 2 request: parent, 10 full bank traces (5 cut with an in-think statement, 5 finished), all positions:
# residual after layers 20 and 21 and the MLP-20 neurons (to reconstruct the MLP-20 input at every token), token log-probs.
import json, random
from wsp.editor import *
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl")]
pc = {p["id"] for p in json.load(open("work/post_commit_cost.json"))}
cutc = [r for r in rows if r["cut"] and r["id"] in pc]; fin = [r for r in rows if not r["cut"]]
rnd = random.Random(2)
pick = rnd.sample(cutc, 5) + rnd.sample(fin, 5)
texts = [trace_text(r) for r in pick]
json.dump([r["id"] for r in pick], open("work/read2_ids.json", "w"))
req = {"type": "read", "model": "original", "texts": texts,
       "return": {"positions": "all", "token_logprobs": True, "hidden_states": [20, 21], "mlp": [20]}}
json.dump(req, open("work/req_read2.json", "w"))
print([r["id"] for r in pick], sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts), "tokens")
