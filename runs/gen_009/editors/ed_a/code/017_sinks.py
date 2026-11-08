# Hypothesis: the gate leak on prompt tokens comes from the first few low-proxy sink positions, whose L23 inputs are identical across texts; projecting the gate rows off them removes it.
import json, numpy as np
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); lab=np.array([r[3] for r in rows]); gid=np.array([r[0] for r in rows])
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); p=np.load('work/proxy.npy')
M=json.load(open('work/fwd_01/meta.json')); off=np.concatenate([[0],np.cumsum(M['n_tokens'])])
for i in range(8):
    V=np.asarray(X[off[:-1]+i],dtype=np.float32)
    print(i,'proxy %.3f..%.3f  spread across texts %.4f  norm %.2f'%((V@p).min(),(V@p).max(),np.linalg.norm(V-V.mean(0),axis=1).max(),np.linalg.norm(V.mean(0))))
# all prompt tokens with low proxy
pr=gid[np.char.startswith(lab,'prompt')]
V=np.asarray(X[pr],dtype=np.float32); pp=V@p
print('prompt tokens proxy<0.8:',(pp<0.8).sum(),'of',len(pp))
th=gid[~np.char.startswith(lab,'prompt')]
rng=np.random.default_rng(0); s=rng.choice(th,20000,replace=False); pt=np.asarray(X[np.sort(s)],dtype=np.float32)@p
print('thinking tokens proxy<0.8:',(pt<0.8).sum(),'of 20000, min %.3f'%pt.min())
S=np.stack([np.asarray(X[off[:-1]+i],dtype=np.float32).mean(0) for i in range(4)],1)
np.save('work/sinks.npy',S)
