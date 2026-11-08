# Holdout requests: generate "both" (parent vs edited) on 2 x 32 fresh practice questions (not in the trace bank), random seed 808.
import json, random
from wsp.editor import questions
Q=questions(); bank={json.loads(l)['id'] for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
ids=sorted(i for i in Q if i not in bank); random.Random(808).shuffle(ids)
for k,nm in enumerate(('p5','p6')):
    json.dump({"type":"generate","model":"both","questions":ids[32*k:32*k+32]},open(f'work/req_{nm}.json','w'))
print(ids[:3], len(ids))
