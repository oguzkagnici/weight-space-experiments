# Hypothesis: none (builds pass-2 request): the edited model teacher-forced on the same 64 DEV traces; next-token distributions shift toward Hence/Therefore only at gated churn breaks.
import json
from wsp.editor import *
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
ids=json.load(open('work/req1_ids.json'))
req={"type":"read","model":"edited","texts":[trace_text(rows[i]) for i in ids],"return":{"token_logprobs":True,"topk":20,"positions":"all"}}
json.dump(req,open('work/req2.json','w')); print(len(ids))
