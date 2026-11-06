# Hypothesis: clipping the new w2 column to the per-row Q8_0 peer-block absmax keeps llama.cpp block scales unchanged while preserving most of the Hence/Therefore-vs-churn readout; the lost norm can be moved into the activation scale.
import torch
from wsp.editor import *
W2o=load_original('model.layers.23.feed_forward.w2.weight'); peers=torch.cat([W2o[:,64:74],W2o[:,75:96]],1); am=peers.abs().max(1).values
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight'); gp=load('model.layers.23.post_ffn_norm.weight')
ids=lambda L:[tok(s,add_special_tokens=False).input_ids[0] for s in L]
up=ids(['Hence','Therefore']); dn=ids(['Alternatively','Maybe','Perhaps','Another','Possibly'])
t=gf*(E[up].mean(0)-E[dn].mean(0)); u=gp*t; u=u/u.norm()
def lens(c):
    lo=E@(gf*(gp*c)); z=(lo-lo.mean())/lo.std()
    return ' '.join(f'{s}:{float(z[i]):+.1f}' for s,i in zip(['Hence','Therefore','Thus','Alternatively','Maybe','Perhaps','Another','Let','But'],ids(['Hence','Therefore','Thus','Alternatively','Maybe','Perhaps','Another','Let','But'])))
print('unclipped:',lens(u))
for K in (1,2,3,5,8):
    c=torch.maximum(torch.minimum(K*u,0.98*am),-0.98*am)
    print(f'K={K}: clipped {(K*u.abs()>0.98*am).sum().item()} rows, ||c|| {c.norm():.3f}, c.u {float(c@u):.3f}, cos {float(c@u/c.norm()):.3f} | {lens(c)}')
