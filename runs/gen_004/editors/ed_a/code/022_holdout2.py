# Hypothesis: (measurement setup) a second, natural-distribution HOLDOUT of 32 fresh practice questions not in the trace bank, for the fixed edit E3.
import json, random
from wsp.editor import questions
bank=set(json.loads(l)['id'] for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl'))
Q=questions(); ids=sorted(q for q in Q if q not in bank)
random.seed(2026); h2=random.sample(ids,32)
json.dump({'holdout2':h2},open('work/holdout2.json','w'))
json.dump({"type":"generate","model":"both","questions":h2},open('work/req_gen_hold2.json','w'))
import collections; print(collections.Counter(Q[q]['category'] for q in h2))
