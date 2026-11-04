# Hypothesis: on 4-option questions v2 should also flatten the A-D letter bias (check letter bias, letter mass, logp split by fresh/used).
import numpy as np, json
from wsp.editor import questions
Q=questions(); s=json.load(open('work/split.json')); used=set(s['dev'][:768])
o={r['id']:r for r in map(json.loads,open('work/fwd_07/score_7_original.jsonl'))}
e={r['id']:r for r in map(json.loads,open('work/fwd_07/score_7_edited.jsonl'))}
ids=[i for i in o if len(o[i]['probs'])==4]
for tag,R in [('orig',o),('v2',e)]:
    P=np.array([R[i]['probs'] for i in ids]); L=np.log(P+1e-12); Lc=L-L.mean(1,keepdims=True)
    print(tag,'4-opt n',len(ids),'bias',np.round(Lc.mean(0),2),'acc',np.mean([R[i]['correct'] for i in ids]),'logp',np.mean([R[i]['logp_correct'] for i in ids]),'mass',np.mean([R[i]['letter_mass'] for i in ids]))
fresh=[i for i in o if i not in used]
print('fresh n',len(fresh),'dacc',np.mean([e[i]['correct']-o[i]['correct'] for i in fresh]),'dlogp',np.mean([e[i]['logp_correct']-o[i]['logp_correct'] for i in fresh]))
