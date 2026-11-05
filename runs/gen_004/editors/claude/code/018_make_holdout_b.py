# HOLDOUT B: 32 more random practice questions, disjoint from the trace bank, DEV and HOLDOUT A.
import json, random
from wsp.editor import *
bank = {json.loads(l)["id"] for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl")}
used = bank | set(json.load(open("work/holdout_ids.json")))
pool = sorted(q for q in questions() if q not in used)
hb = random.Random("claude-gen4-holdout-b").sample(pool, 32)
json.dump(hb, open("work/holdout_b_ids.json", "w"))
json.dump({"type": "generate", "model": "both", "questions": hb}, open("work/req_gen_holdout_b.json", "w"))
print(len(hb))
