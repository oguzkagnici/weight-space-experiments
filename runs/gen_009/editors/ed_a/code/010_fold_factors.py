# Hypothesis: n5105 and n881 are scaled copies of n331 (w1 ratio a ~1, w3 ratio b), so h_j ~ a*b*h_331 and w2[:,331] += a*b*w2[:,j] frees them.
import torch, json
from wsp.editor import *
w1=load('model.layers.23.feed_forward.w1.weight'); w3=load('model.layers.23.feed_forward.w3.weight'); w2=load('model.layers.23.feed_forward.w2.weight')
k=331; out={}
for j in [5105,881]:
    a=float(w1[j]@w1[k]/(w1[k]@w1[k])); b=float(w3[j]@w3[k]/(w3[k]@w3[k]))
    r1=float((w1[j]-a*w1[k]).norm()/w1[j].norm()); r3=float((w3[j]-b*w3[k]).norm()/w3[j].norm())
    out[j]=a*b; print(j,'a %.4f b %.4f s %.4f resid w1 %.4f w3 %.4f'%(a,b,a*b,r1,r3))
# Q8 block absmax for the free slots (excluding the slot and 331)
for j in [5105,881]:
    b0=(j//32)*32; cols=[c for c in range(b0,b0+32) if c not in (j,331)]
    am=w2[:,cols].abs().max(dim=1).values
    print(j,'block',b0,'absmax per row: median %.4f min %.4f'%(am.median(),am.min()),'|clip vec| %.3f'%(0.98*am).norm())
json.dump(out,open('work/fold.json','w'))
