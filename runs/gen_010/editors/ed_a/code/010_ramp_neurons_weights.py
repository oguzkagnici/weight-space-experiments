# Hypothesis: a few late-layer MLP neurons write an "early letters up, late letters down" ramp (content-free letter prior); find them from w2 columns read through the letter unembeddings.
import numpy as np, torch
from wsp.editor import *
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy(); g=load('model.embedding_norm.weight').numpy()
Ec=E-E.mean(0)
r=np.arange(10)-4.5; r/=np.linalg.norm(r)
out={}
for l in range(14,24):
    w2=load(f'model.layers.{l}.feed_forward.w2.weight').numpy(); pg=load(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    V=(Ec*g)@(pg[:,None]*w2)   # 10 x 7680 letter-logit write per unit activation
    ramp=r@V                  # projection on ramp (negative = early up)
    out[l]=V
    idx=np.argsort(ramp)[:5]
    print(l, 'most early-up', [(int(i), round(float(ramp[i]),3), round(float(np.abs(V[:,i]).max()),3)) for i in idx])
np.savez('work/letterwrite_V.npz', **{str(k):v for k,v in out.items()})
