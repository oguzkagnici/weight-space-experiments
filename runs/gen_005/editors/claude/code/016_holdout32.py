# HOLDOUT32: 32 trace-bank questions not in DEV32, chosen by a fixed seed (no look at results), for the -15 dose
# (my edit is still the -15 dose from code/014).
import json, random
from wsp.editor import *
bank = [json.loads(l)["id"] for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")]
dev = set(json.load(open("work/dev32.json")))
rest = sorted(q for q in bank if q not in dev)
ho = sorted(random.Random("claude-gen5-holdout").sample(rest, 32))
json.dump(ho, open("work/holdout32.json", "w"))
json.dump({"type": "generate", "model": "both", "questions": ho}, open("work/req_gen_ho.json", "w"))
print(edited(), len(ho))
