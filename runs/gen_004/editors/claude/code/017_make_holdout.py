# HOLDOUT question set for the confirmation pass: 32 random practice questions never used by this editor (not in the trace bank).
import json, random
from wsp.editor import *
bank = {json.loads(l)["id"] for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl")}
pool = sorted(q for q in questions() if q not in bank)
hold = random.Random("claude-gen4-holdout").sample(pool, 32)
json.dump(hold, open("work/holdout_ids.json", "w"))
json.dump({"type": "generate", "model": "both", "questions": hold}, open("work/req_gen_holdout.json", "w"))
print(len(hold), "HOLDOUT questions (not looked at)")
