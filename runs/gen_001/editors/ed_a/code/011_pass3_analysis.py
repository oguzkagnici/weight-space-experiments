# Hypothesis: the bias-neuron projection edit reduced the constant letter bias; check remaining bias, where it lives (per-layer component means, orig vs edited) and how much downstream layers compensated.
from wsp.editor import *
import numpy as np, json, glob
def load(f): return {r['id']:r for r in map(json.loads,open(f))}
o=load('work/fwd_03/score_3_original.jsonl'); e=load('work/fwd_03/score_3_edited.jsonl')
ids=list(json.load(open('work/fwd_03/score_3_acts/original/index.json'))['ids']) if glob.glob('work/fwd_03/score_3_acts/original/index.json') else list(o)
Q=questions(); nopt=np.array([len(Q[i]['options']) for i in ids]); m10=nopt==10
def bias(R):
    Lx=np.log(np.array([R[i]['probs'] for i in np.array(ids)[m10]])+1e-12); Lx-=Lx.mean(1,keepdims=True); return Lx.mean(0)
print('orig bias',np.round(bias(o),2)); print('edit bias',np.round(bias(e),2))
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
for m in ['original','edited']:
    D=f'work/fwd_03/score_3_acts/{m}/'
    h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
    out=[]
    for nm in ['h20']+[f'out_model.layers.{l}.post_operator_norm' for l in [21,22,23]]+[f'out_model.layers.{l}.post_ffn_norm' for l in [20,21,22,23]]:
        x=np.load(D+nm+'.npy').astype(np.float32); c=((x@Uc)/r)[m10].mean(0)
        print(m, nm.replace('out_model.layers.','L'), np.round(c,2))
