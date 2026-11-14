# Hypothesis H2: key inputs of option-line tokens carry an ordinal slot direction; record k_proj inputs at all positions of 20 thinking-mode prompts (questions outside all banks) to estimate it.
import json, numpy as np
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
used=set(json.load(open(S+'holdout48_ids.json')))|set(json.load(open(S+'holdout_bank_ids.json')))
for f in ('trace_bank_gen008champ_cap8192_det1.jsonl',): used|={json.loads(l)['id'] for l in open(S+f)}
Q=questions(); rng=np.random.default_rng(12)
cand=[q for q in sorted(Q) if q not in used and len(Q[q]['options'])==10]
rng.shuffle(cand)
sel=[];tot=0
for q in cand:
    n=len(tok.encode(render_think(q)))
    if n<=330: sel.append(q); tot+=n
    if len(sel)==22: break
print(len(sel),tot,[Q[q]['category'] for q in sel])
AL=[2,4,6,8,10,12,14,16,18,20,22]
req={"type":"read","model":"original","texts":[render_think(q) for q in sel],
     "return":{"inputs":[f"model.layers.{l}.self_attn.k_proj" for l in AL],"positions":"all"}}
json.dump(req,open('work/req_p2.json','w')); json.dump(sel,open('work/p2_ids.json','w'))
