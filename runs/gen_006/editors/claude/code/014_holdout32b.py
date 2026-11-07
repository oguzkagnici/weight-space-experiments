# HOLDOUT32-B: a second set of 32 fresh practice questions (not in the bank, not in HOLDOUT32), seed
# "claude-gen6-holdout-b", to double the holdout evidence for the unchanged edit v1 (same hypothesis as code/012).
import json, random
from wsp.editor import questions
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
bank = {json.loads(l)["id"] for l in open(BANK)}
ho_a = set(json.load(open("work/holdout32.json"))["holdout32"])
pool = sorted(q for q in questions() if q not in bank and q not in ho_a)
ho = sorted(random.Random("claude-gen6-holdout-b").sample(pool, 32))
json.dump({"holdout32b": ho}, open("work/holdout32b.json", "w"))
json.dump({"type": "generate", "model": "both", "questions": ho}, open("work/req_06.json", "w"))
print(len(pool), ho[:4])
