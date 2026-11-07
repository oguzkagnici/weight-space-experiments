# Hypothesis: freeing duplicate neuron j by doubling its twin k (exact x2 in bf16/Q8) leaves a residual h_j-h_k far below a quiet neuron's impact; quantify residual and Q8 block-scale side effects (up-row x2 vs w2-col x2).
import numpy as np, torch
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); T=H.shape[0]
Hs=np.asarray(H[::3],dtype=np.float32)
W2=load('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float().numpy()
imp=np.load('work/l23_impact.npy'); rmsh=np.load('work/l23_rmsh.npy'); spread=imp/rmsh
A=np.abs(Hs)
for k,j,s in [(1228,3401,1),(1228,6986,-1),(1228,3793,-1),(3793,6986,1),(331,5105,1),(331,881,1),(5105,881,1),(5821,2400,1)]:
    # the sign s: h_j ~ s*h_k ; w2_j ~ s*w2_k so contribution w2_j h_j ~ w2_k h_k
    r=Hs[:,j]-s*Hs[:,k]
    resid_imp=np.sqrt((r**2).mean())*spread[j]
    bk=k//32; others=[i for i in range(bk*32,bk*32+32) if i not in (k,j)]
    om=A[:,others].max(1)
    frac_up_now=(A[:,k]>om).mean(); frac_up2=(2*A[:,k]>om).mean()
    bj=j//32; oj=[i for i in range(bj*32,bj*32+32) if i!=j]
    frac_jmax=(A[:,j]>A[:,oj].max(1)).mean()
    rb=np.abs(W2[:,bk*32:bk*32+32]); oth=np.delete(rb,[k-bk*32]+([j-bk*32] if j//32==bk else []),axis=1).max(1)
    rows_w2=(2*np.abs(W2[:,k])>oth).sum(); rows_now=(np.abs(W2[:,k])>oth).sum()
    rbj=np.abs(W2[:,bj*32:bj*32+32]); othj=np.delete(rbj,[j-bj*32],axis=1).max(1); rows_jmax=(np.abs(W2[:,j])>othj).sum()
    print(f'free {j} into {k} (s={s:+d}): resid rms {np.sqrt((r**2).mean()):.4f} mean {r.mean():+.4f} resid_imp {resid_imp:.2e} (quietest neuron 2e-4) | act-block max: k now {frac_up_now:.3f} k*2 {frac_up2:.3f}, j now {frac_jmax:.3f} | w2 rows k max now {rows_now} k*2 {rows_w2}; j max rows {rows_jmax}')
