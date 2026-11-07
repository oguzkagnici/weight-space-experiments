# Hypothesis: the near-duplicate L23 neurons (corr 0.9999) have (near-)identical w1 rows and proportional w3 rows, so h_j = lambda*h_k follows from the weights (no fitting).
import numpy as np, torch
from wsp.editor import *
W1=load('model.layers.23.feed_forward.w1.weight'); W3=load('model.layers.23.feed_forward.w3.weight'); W2=load('model.layers.23.feed_forward.w2.weight')
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r')
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
sub=np.asarray(X[::7],dtype=np.float32); Hs=np.asarray(H[::7],dtype=np.float32)
imp=np.load('work/l23_impact.npy'); rmsh=np.load('work/l23_rmsh.npy')
def c(a,b): return float((a@b)/(a.norm()*b.norm()))
for grp in [[1228,3401,6986,3793],[331,5105,881],[5821,2400]]:
    k=grp[0]
    for j in grp[1:]:
        g_j=sub@W1[j].numpy(); g_k=sub@W1[k].numpy(); u_j=sub@W3[j].numpy(); u_k=sub@W3[k].numpy()
        print(f'{j}~{k}: cos w1 {c(W1[j],W1[k]):.4f} cos w3 {c(W3[j],W3[k]):.4f} cos w2 {c(W2[:,j],W2[:,k]):.4f} | gate mean {g_j.mean():.2f}/{g_k.mean():.2f} sd {g_j.std():.2f}/{g_k.std():.2f} up mean {u_j.mean():.2f}/{u_k.mean():.2f} sd {u_j.std():.2f} | h mean {Hs[:,j].mean():.3f}/{Hs[:,k].mean():.3f} sd {Hs[:,j].std():.3f} | w1norm {W1[j].norm():.3f}/{W1[k].norm():.3f} w3norm {W3[j].norm():.3f}/{W3[k].norm():.3f}')
