# Hypothesis: (read, edited) at eval sweep ends P(Hence) -> ~1, after Hence P(" answer") -> ~1, and off-target token log-probs barely change.
import json
D=json.load(open('work/pass1_plan.json'))
idx=[i for i,m in enumerate(D['meta']) if m['kind']=='probe']+[i for i,m in enumerate(D['meta']) if m['kind']=='full' and not m['cut']]
print(len(idx),sum(D['ntok'][i] for i in idx))
json.dump(dict(idx=idx),open('work/pass2_idx.json','w'))
req={"type":"read","model":"edited","texts":[D['texts'][i] for i in idx],"return":{"token_logprobs":True,"topk":20,"positions":"all"}}
json.dump(req,open('work/req2.json','w'))
