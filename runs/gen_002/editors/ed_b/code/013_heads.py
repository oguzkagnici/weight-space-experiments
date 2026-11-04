# Hypothesis: specific attention heads in layers 8-14 write the option-end correctness signal (along d14) with high SNR; they are candidates to amplify.
from wsp.editor import *
import numpy as np, json, torch
recs=json.load(open('work/recs2.json')); Z=np.load('work/p2_index.npz'); idx,g,y=Z['idx'],Z['g'],Z['y']
gold=np.array([r['gold'] for r in recs]); nq=len(recs)
d=np.load('work/d14_p1.npy'); d=d/np.linalg.norm(d)
h14=np.load('work/p2_end14.npy').astype(np.float64); rms14=np.sqrt((h14**2).mean(1))
def center(s):
    s=s.copy()
    for qi in range(nq): s[g==qi]-=s[g==qi].mean()
    return s
def stats(s):
    s=center(s); acc=np.mean([np.argmax(s[g==qi])==gold[qi] for qi in range(nq)])
    return (s[y==1].mean()-s[y==0].mean()), s[y==0].std(), acc
tot=center((h14/rms14[:,None])@d); print('total h14 proj: diff %.3f sd %.3f acc %.3f'%stats(tot))
res={}
for l in [8,10,12,14]:
    Zin=np.asarray(np.load(f'work/fwd_02/in_model.layers.{l}.self_attn.o_proj.npy',mmap_mode='r')[idx],dtype=np.float64)
    Wo=load(f'model.layers.{l}.self_attn.o_proj.weight').double().numpy(); wp=load(f'model.layers.{l}.post_operator_norm.weight').double().numpy()
    O=Zin@Wo.T; r=np.sqrt((O**2).mean(1)+1e-6)
    dd=wp*d  # project post-normed contribution
    full=(O/r[:,None])@dd/rms14
    print(f'L{l} attn total: diff %.3f sd %.3f acc %.3f'%stats(full))
    for h in range(16):
        c=(Zin[:,h*128:(h+1)*128]@Wo[:,h*128:(h+1)*128].T)
        s=(c/r[:,None])@dd/rms14
        df,sd,acc=stats(s); res[(l,h)]=(df,sd,acc)
    top=sorted(range(16),key=lambda h:-res[(l,h)][0])[:5]
    print('   top heads:',' '.join(f'H{h} diff {res[(l,h)][0]:.3f} sd {res[(l,h)][1]:.3f} snr {res[(l,h)][0]/res[(l,h)][1]:.2f} acc {res[(l,h)][2]:.2f}' for h in top))
json.dump({f'{k[0]}_{k[1]}':v for k,v in res.items()},open('work/head_stats.json','w'))
