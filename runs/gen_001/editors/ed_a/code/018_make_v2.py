# Hypothesis: v2 = v1 (12 bias neurons) + 15 further bias-aligned low-content neurons (L21-23) + head L22h6 (near-constant letter-bias head), each with the centered letter-unembedding subspace projected out of its output (in post-norm-scaled space), removes more of the early-letter bias (direct estimate: logp +0.085 over v1).
import numpy as np, torch
from wsp.editor import *
V1=[(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820)]
ADD=[(22,4543),(21,218),(23,6518),(23,4531),(22,5653),(23,514),(23,3073),(23,2189),(21,4481),(22,2212),(22,3998),(23,1439),(23,7466),(21,3455),(21,4902)]
HEADS=[(22,6)]
for n in list(edited()): drop_edit(n)
E=load_original('model.embed_tokens.weight')[LETTER_IDS]; wn=load_original('model.embedding_norm.weight')
U=(wn*E).T; Uc=U-U.mean(1,keepdim=True)
Qb,_=torch.linalg.qr(Uc); Qb=Qb[:,:9]
W={}
def get(k):
    if k not in W: W[k]=load_original(k).clone()
    return W[k]
for l,n in V1+ADD:
    w=get(f'model.layers.{l}.feed_forward.w2.weight'); pw=load_original(f'model.layers.{l}.post_ffn_norm.weight')
    v=pw*w[:,n]; w[:,n]=(v-Qb@(Qb.T@v))/pw
for l,h in HEADS:
    w=get(f'model.layers.{l}.self_attn.o_proj.weight'); pw=load_original(f'model.layers.{l}.post_operator_norm.weight')
    sl=slice(128*h,128*h+128); V=pw[:,None]*w[:,sl]; V2=V-Qb@(Qb.T@V); w[:,sl]=V2/pw[:,None]
    print('head',l,h,'rel change',float((V-V2).norm()/V.norm()))
for k,t in W.items(): print(save_edit(k,t))
