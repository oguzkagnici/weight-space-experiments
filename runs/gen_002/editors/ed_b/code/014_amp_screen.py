# Hypothesis: amplifying high-SNR correctness-writing heads (L12.H13, L10.H4, L8.H1) raises option-end belief accuracy (linear pre-screen along d14 at h14 and h16).
from wsp.editor import *
import numpy as np, json
recs=json.load(open('work/recs2.json')); Z=np.load('work/p2_index.npz'); idx,g,y=Z['idx'],Z['g'],Z['y']
gold=np.array([r['gold'] for r in recs]); nq=len(recs); h12=np.array([r['h12'] for r in recs])
d=np.load('work/d14_p1.npy'); d=d/np.linalg.norm(d)
def center(s):
    s=s.copy()
    for qi in range(nq): s[g==qi]-=s[g==qi].mean()
    return s
def acc(s): return np.mean([np.argmax(s[g==qi])==gold[qi] for qi in range(nq)])
comp={}
for lay in [14,16]:
    H=np.load(f'work/p2_end{lay}.npy').astype(np.float64); rms=np.sqrt((H**2).mean(1))
    tot=center((H/rms[:,None])@d)
    for (l,h) in [(12,13),(10,4),(8,1),(12,8)]:
        Zin=np.asarray(np.load(f'work/fwd_02/in_model.layers.{l}.self_attn.o_proj.npy',mmap_mode='r')[idx],dtype=np.float64)
        Wo=load(f'model.layers.{l}.self_attn.o_proj.weight').double().numpy(); wp=load(f'model.layers.{l}.post_operator_norm.weight').double().numpy()
        O=Zin@Wo.T; r=np.sqrt((O**2).mean(1)+1e-6)
        s=center(((Zin[:,h*128:(h+1)*128]@Wo[:,h*128:(h+1)*128].T)/r[:,None])@(wp*d)/rms)
        c=np.corrcoef(s,tot-s)[0,1]
        print(f'h{lay} L{l}H{h}: corr(head,rest) {c:.2f} | '+' '.join(f'x{1+b}: {acc(tot+b*s):.3f}' for b in [-1,0,0.5,1,2,4]))
