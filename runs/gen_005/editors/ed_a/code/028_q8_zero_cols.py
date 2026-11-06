# Hypothesis: every early divergence in passes 3-4 (same positions for v2 and v3) comes from removing the original n74's small contribution; a neuron whose w2 column already quantises to (almost) all zeros in Q8_0 can be repurposed with no effect at non-firing positions in llama.cpp.
import torch, numpy as np
from wsp.editor import *
for L in (23,22,21,20):
    W=load_original(f'model.layers.{L}.feed_forward.w2.weight').bfloat16().float()   # 2048 x 7680
    B=W.view(2048,240,32); d=B.abs().amax(-1,keepdim=True)/127
    q=torch.round(B/d).view(2048,7680)
    nz=(q!=0).sum(0)          # per column: rows with nonzero quantised weight
    o=torch.argsort(nz)[:6]
    print(L,'min nonzero rows per column:',[(int(i),int(nz[i])) for i in o],' median',int(nz.median()))
