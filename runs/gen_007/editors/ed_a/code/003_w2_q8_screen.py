# Hypothesis: some MLP neurons (in some layer) have w2 columns that quantise (Q8_0, blocks of 32 along the neuron axis) to all-zero, i.e. contribute exactly nothing in llama.cpp -> free slots for silent edits.
import torch, numpy as np
from wsp.editor import *
res=[]
for L in range(24):
    W=load(f"model.layers.{L}.feed_forward.w2.weight").to(torch.bfloat16).float()  # 2048 x 7680
    A=W.abs().view(2048,240,32)
    amax=A.amax(-1,keepdim=True)
    q=torch.round(A/(amax/127)).view(2048,7680)
    nz=(q>0).sum(0)            # nonzero quantised entries per neuron column
    ismax=(A==amax).view(2048,7680).sum(0)
    cn=W.norm(dim=0)
    w1=load(f"model.layers.{L}.feed_forward.w1.weight").norm(dim=1); w3=load(f"model.layers.{L}.feed_forward.w3.weight").norm(dim=1)
    o=torch.argsort(nz)[:3]
    print(L,'min nonzero entries',[ (int(j),int(nz[j]),int(ismax[j]),round(float(cn[j]),3)) for j in o], 'median nz',int(nz.median()),'colnorm med',round(float(cn.median()),3),
          'w1 min',round(float(w1.min()),3),'w3 min',round(float(w3.min()),3))
    np.save(f'work/w2nz_L{L}.npy',nz.numpy())
