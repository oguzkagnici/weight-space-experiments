# Hypothesis: check how much the L22 suppressor x0.5 edit actually moved letter logits vs the direct-effect prediction (self-repair downstream?).
import numpy as np, json, glob
def load(f):
    R=[json.loads(l) for l in open(f)]; return {r['id']:r for r in R}
o=load(glob.glob('work/fwd_02/*original.jsonl')[0]); e=load(glob.glob('work/fwd_02/*edited.jsonl')[0])
ids=[i for i in o if len(o[i]['probs'])==10]
Lo=np.log(np.array([o[i]['probs'] for i in ids])+1e-12); Le=np.log(np.array([e[i]['probs'] for i in ids])+1e-12)
Lo-=Lo.mean(1,keepdims=True); Le-=Le.mean(1,keepdims=True)
print('orig bias',np.round(Lo.mean(0),2)); print('edit bias',np.round(Le.mean(0),2))
d=Le-Lo
po=Lo.argmax(1)
print('mean change of orig-top letter',d[np.arange(len(ids)),po].mean(), ' others',(d.sum(1)-d[np.arange(len(ids)),po]).mean()/9)
print('mean |change|',np.abs(d).mean())
