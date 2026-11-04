# Hypothesis: anti-correct writers at option ends (L12.H12, L10.H6) dilute the belief; removing them raises belief accuracy (lens screen at h14/h16 along d14).
from wsp.editor import *
import numpy as np, json
recs=json.load(open('work/recs2.json')); Z=np.load('work/p2_index.npz'); idx,g,y=Z['idx'],Z['g'],Z['y']
gold=np.array([r['gold'] for r in recs]); nq=len(recs)
d=np.load('work/d14_p1.npy'); d=d/np.linalg.norm(d)
def center(s):
    s=s.copy()
    for qi in range(nq): s[g==qi]-=s[g==qi].mean()
    return s
def acc(s): return np.mean([np.argmax(s[g==qi])==gold[qi] for qi in range(nq)])
def sep(s): return (s[y==1].mean()-s[y==0].mean())/s[y==0].std()
cache={}
for l in [10,12]:
    Zin=np.asarray(np.load(f'work/fwd_02/in_model.layers.{l}.self_attn.o_proj.npy',mmap_mode='r')[idx],dtype=np.float64)
    Wo=load(f'model.layers.{l}.self_attn.o_proj.weight').double().numpy(); wp=load(f'model.layers.{l}.post_operator_norm.weight').double().numpy()
    O=Zin@Wo.T; r=np.sqrt((O**2).mean(1)+1e-6)
    for h in range(16): cache[(l,h)]=((Zin[:,h*128:(h+1)*128]@Wo[:,h*128:(h+1)*128].T)/r[:,None])@(wp*d)
for lay in [14,16]:
    H=np.load(f'work/p2_end{lay}.npy').astype(np.float64); rms=np.sqrt((H**2).mean(1)); tot=center((H/rms[:,None])@d)
    hs={kk:center(v/rms) for kk,v in cache.items()}
    for name,ch in [('base',0),('-L12H12',-hs[(12,12)]),('-L10H6',-hs[(10,6)]),('-both',-hs[(12,12)]-hs[(10,6)]),('-both +L12H13',-hs[(12,12)]-hs[(10,6)]+hs[(12,13)])]:
        s=tot+ch; print(f'h{lay} {name:16s} acc {acc(s):.3f} sep {sep(s):.3f}')
    print('corr L12H12 vs rest',np.corrcoef(hs[(12,12)],tot-hs[(12,12)])[0,1])
