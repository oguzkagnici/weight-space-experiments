# Hypothesis: (1) scaling down the L22 letter-suppressor neurons' w2 columns by alpha reduces the letter-asymmetric suppression of the retrieved answer; (2) projecting the centered letter-unembedding subspace out of a few near-constant bias neurons' outputs removes a fixed early-letter bias. Usage: py code/008_make_edit.py <alpha_sup22> <n_bias> [alpha_sup23]
import sys, numpy as np, torch
from wsp.editor import *
alpha=float(sys.argv[1]); nb=int(sys.argv[2]); a23=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
SUP22=[389,5859,6620,1812,464,7357,6492,3344,397,2786,3310]
SUP23=[3631,7209,1192,1876,2758,3520,1107,358,6512,1041]
BIAS=[(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612)][:nb]
for n in list(edited()): drop_edit(n)
E=load_original('model.embed_tokens.weight')[LETTER_IDS]; wn=load_original('model.embedding_norm.weight')
U=(wn*E).T; Uc=U-U.mean(1,keepdim=True)          # 2048x10
Q_,_=torch.linalg.qr(Uc)                           # orthonormal basis (rank 9 effectively)
Q_=Q_[:, :9]
W={}
def get(l):
    k=f'model.layers.{l}.feed_forward.w2.weight'
    if k not in W: W[k]=load_original(k).clone()
    return W[k]
if alpha!=1.0:
    w=get(22); w[:,SUP22]*=alpha
if a23!=1.0:
    w=get(23); w[:,SUP23]*=a23
for l,n in BIAS:
    w=get(l); pw=load_original(f'model.layers.{l}.post_ffn_norm.weight')
    v=pw*w[:,n]
    before=(v@Uc).numpy()
    v2=v-Q_@(Q_.T@v)
    w[:,n]=v2/pw
    print(l,n,'letter logits before',np.round(before,2),'after',np.round((v2@Uc).numpy(),3), 'rel change', float((v-v2).norm()/v.norm()))
for k,t in W.items(): print(k, save_edit(k,t))
print(edited())
