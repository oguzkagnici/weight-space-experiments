# Hypothesis H1 prerequisite: late MLP w2 (and w1,w3) have a long tail of small singular values; save SVDs of w2 for L16-23.
import torch, numpy as np, time
from wsp.editor import *
torch.set_num_threads(3)
for l in range(16,24):
    for m in ('w2',):
        W=load_original(f'model.layers.{l}.feed_forward.{m}.weight')
        t=time.time(); U,S,Vh=torch.linalg.svd(W,full_matrices=False)
        e=(S**2).cumsum(0)/(S**2).sum()
        print(l,m,tuple(W.shape),'s0 %.2f s_med %.3f s_min %.3f'%(S[0],S[len(S)//2],S[-1]),'energy@64/256/512/1024: '+' '.join('%.3f'%e[k-1] for k in (64,256,512,1024)),'%.0fs'%(time.time()-t),flush=True)
        np.save(f'work/svd_L{l}_{m}_U.npy',U.numpy()); np.save(f'work/svd_L{l}_{m}_S.npy',S.numpy()); np.save(f'work/svd_L{l}_{m}_Vh.npy',Vh.numpy().astype(np.float16))
