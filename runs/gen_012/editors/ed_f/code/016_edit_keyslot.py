# Hypothesis H2 edit: remove the ordinal slot subspace (slope + first-item directions of slot means, pass 2) from the input side of k_proj in chosen attention layers: W_k <- W_k (I - U U^T).
import sys, json, numpy as np, torch
from wsp.editor import *
layers=[int(x) for x in sys.argv[1].split(',')]; rank=int(sys.argv[2]); alpha=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
for n in list(edited()): drop_edit(n)
for l in layers:
    m=np.load(f'work/slot_means_L{l}.npy')              # 10 x 2048 centred slot means
    slope=(np.arange(10)-4.5)@m; first=m[0]-m[1:].mean(0)
    B=np.stack([slope,first][:rank],1); U,_=np.linalg.qr(B); U=torch.tensor(U,dtype=torch.float32)
    name=f'model.layers.{l}.self_attn.k_proj.weight'; W=load_original(name)
    W2=W-alpha*(W@U)@U.T
    rel=((W2-W).norm()/W.norm()).item()
    print(l,tuple(W.shape),'rel change %.4f'%rel, save_edit(name,W2))
print(edited())
