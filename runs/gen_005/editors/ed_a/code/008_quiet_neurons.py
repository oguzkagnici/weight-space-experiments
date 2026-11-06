# Hypothesis: some MLP neurons in L20/L23 have tiny output columns (after post_ffn_norm scaling) and can be repurposed without losing function.
import torch, numpy as np
from wsp.editor import *
for L in (20,21,22,23):
    w2=load(f'model.layers.{L}.feed_forward.w2.weight'); g=load(f'model.layers.{L}.post_ffn_norm.weight')
    w1=load(f'model.layers.{L}.feed_forward.w1.weight'); w3=load(f'model.layers.{L}.feed_forward.w3.weight')
    cn=(w2).norm(dim=0); r1=w1.norm(dim=1); r3=w3.norm(dim=1)
    o=torch.argsort(cn)[:8]
    print(L,'w2 col norm median %.3f min'%cn.median(), [(int(i),round(float(cn[i]),3),round(float(r1[i]),2),round(float(r3[i]),2)) for i in o])
    print('   gpost mean %.3f'%g.mean(), 'w1 row norm median %.2f w3 %.2f'%(r1.median(),r3.median()))
    if L==20: print('   n5414', float(cn[5414]), float(r1[5414]), float(r3[5414]))
