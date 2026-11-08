# Hypothesis: unit A fires inside the prompt at the end of the option list (a listing "sweep end"); find where and how often.
import json, numpy as np
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); D=json.load(open('work/pass1_plan.json'))
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
M=json.load(open('work/fwd_01/meta.json')); off=np.concatenate([[0],np.cumsum(M['n_tokens'])])
w1=load('model.layers.23.feed_forward.w1.weight').numpy(); w3=load('model.layers.23.feed_forward.w3.weight').numpy()
silu=lambda v: v/(1+np.exp(-v))
pr=[r for r in rows if r[3].startswith('prompt')]
g=np.array([r[0] for r in pr]); V=np.asarray(X[g],dtype=np.float32)
for j in [5105,881]:
    pre=V@w1[j]; h=silu(pre)*(V@w3[j])
    o=np.argsort(-np.abs(h))[:8]
    for i in o:
        r=pr[i]; ti=r[1]; t=D['texts'][ti]; enc=tok(t,return_offsets_mapping=True,add_special_tokens=False)['offset_mapping']; k=r[0]-off[ti]
        c=enc[k][1]
        print(j,'h %.3f pre %.2f'%(h[i],pre[i]),r[2],repr(t[max(0,c-60):c]),'>>',repr(t[c:c+30]))
    print(j,'prompt tokens with |h|>0.005:',(np.abs(h)>0.005).sum(),'texts:',len(set(pr[i][1] for i in np.nonzero(np.abs(h)>0.005)[0])))
