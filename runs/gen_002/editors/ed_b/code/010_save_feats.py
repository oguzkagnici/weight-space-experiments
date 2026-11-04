# Hypothesis: none (save compact option-end / last-position features from pass 1, then the big arrays can be deleted).
import numpy as np, json
recs=json.load(open('work/recs1.json'))
out={}
for lay in [14,18,20,22,24]:
    H=np.load(f'work/fwd_01/h{lay}.npy',mmap_mode='r')
    out[f'end{lay}']=np.concatenate([np.asarray(H[[r['start']+p for p in r['ends']]]) for r in recs]).astype(np.float32)
    out[f'last{lay}']=np.stack([np.asarray(H[r['start']+r['n']-1]) for r in recs]).astype(np.float32)
K=np.load('work/fwd_01/out_model.layers.22.self_attn.k_norm.npy',mmap_mode='r')
out['kend22']=np.concatenate([np.asarray(K[[r['start']+p for p in r['ends']]]) for r in recs]).astype(np.float32)
np.savez('work/p1_feats.npz',**out); print({k:v.shape for k,v in out.items()})
