# HOLDOUT32: 32 fresh practice questions never used here (not in the trace bank), seed "claude-gen6-holdout".
# Hypothesis to test there: the finalist gate's gain on DEV32 carries over to random questions (expected small: only
# about 1 in 5 questions narrows to two finalists and is cut).
import json, random
from wsp.editor import questions
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
bank = {json.loads(l)["id"] for l in open(BANK)}
pool = sorted(q for q in questions() if q not in bank)
ho = sorted(random.Random("claude-gen6-holdout").sample(pool, 32))
json.dump({"holdout32": ho}, open("work/holdout32.json", "w"))
json.dump({"type": "generate", "model": "both", "questions": ho}, open("work/req_05.json", "w"))
print(len(pool), "pool;", ho[:5])
