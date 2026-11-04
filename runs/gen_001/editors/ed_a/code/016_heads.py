# Hypothesis: some attention heads write a content-independent letter bias at the answer position (low-variance letter contribution, little correct-letter content); per-head direct logit attribution from o_proj inputs (pass 1).
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions(); nopt=np.array([len(Q[r['id']]['options']) for r in rows]); m10=nopt==10; ans=np.array([r['answer'] for r in rows])
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
b=np.array([1.15,1.66,.57,.16,.03,1.27,-.05,-1.14,-2.09,-1.58]); b-=b.mean(); bn=b/np.linalg.norm(b)
out=[]
for l in [2,4,6,8,10,12,14,16,18,20,22]:
    z=np.load(D+f'in_model.layers.{l}.self_attn.o_proj.npy').astype(np.float32)  # n x 2048
    Wo=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_operator_norm.weight').numpy()
    x=z@Wo.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
    for h in range(16):
        sl=slice(128*h,128*h+128)
        c=((z[:,sl]@Wo[:,sl].T)*pw)@Uc*s[:,None]   # n x10
        mb=c[m10].mean(0); var=c[m10].std(0).mean()
        cont=c[np.arange(len(ans)),ans].mean()
        out.append((mb@bn,l,h,cont,var,np.round(mb,2)))
out.sort(key=lambda x:-abs(x[0]))
for o in out[:14]: print(f'L{o[1]}h{o[2]} biasproj {o[0]:.3f} content {o[3]:.3f} sd {o[4]:.3f}',o[5])
