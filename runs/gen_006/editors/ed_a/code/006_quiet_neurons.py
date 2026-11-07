# Hypothesis: some MLP neurons have w2 columns that quantise (Q8_0) to all or nearly all zeros, so repurposing them leaves every thinking token bit-identical.
import torch, numpy as np
from wsp.editor import load
res={}
for L in range(24):
    W=load(f'model.layers.{L}.feed_forward.w2.weight').bfloat16().float()   # [2048,7680]
    Wb=W.view(2048,240,32)
    d=Wb.abs().amax(-1,keepdim=True)/127
    q=torch.round(Wb/d.clamp_min(1e-30)).view(2048,7680)
    nz=(q!=0).sum(0)             # nonzero quantised entries per neuron column
    cn=W.norm(dim=0)
    w1=load(f'model.layers.{L}.feed_forward.w1.weight'); w3=load(f'model.layers.{L}.feed_forward.w3.weight')
    o=torch.argsort(nz)[:5]
    print(L,'min nonzero q entries per column:',[(int(i),int(nz[i]),round(float(cn[i]),4)) for i in o],'median col norm %.3f'%cn.median(), 'w1n %.2f w3n %.2f'%(w1[o[0]].norm(),w3[o[0]].norm()))
    res[L]=nz.numpy()
np.save('work/w2_q8_nonzero.npy',np.stack([res[L] for L in range(24)]))
