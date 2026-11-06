# Hypothesis: some L23 neurons with small output columns are also nearly silent along the traces (|act|*||w2 col|| tiny), so repurposing one costs nothing; and the mean MLP-input direction is a usable near-constant bias proxy.
import json, numpy as np, torch
from wsp.editor import *
D='work/fwd_01'
X=np.load(f'{D}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
rng=np.random.default_rng(2); samp=np.sort(rng.choice(X.shape[0],40000,replace=False))
Xs=torch.tensor(np.asarray(X[samp],np.float32))
w1=load('model.layers.23.feed_forward.w1.weight'); w3=load('model.layers.23.feed_forward.w3.weight'); w2=load('model.layers.23.feed_forward.w2.weight')
gp=load('model.layers.23.post_ffn_norm.weight')
cn=(w2*gp[:,None]).norm(dim=0)
A=torch.nn.functional.silu(Xs@w1.T)*(Xs@w3.T)   # 40000 x 7680
contrib=(A.abs().mean(0))*cn
mx=(A.abs().max(0).values)*cn
o=torch.argsort(contrib)[:12]
print('median mean|a|*||gp*w2col|| %.4f'%contrib.median())
for i in o: print(int(i),'mean|a| %.3f max|a| %.2f gpcol %.3f contrib %.5f maxcontrib %.4f'%(A[:,i].abs().mean(),A[:,i].abs().max(),cn[i],contrib[i],mx[i]))
mu=Xs.mean(0); mh=mu/mu.norm(); p=Xs@mh
print('bias proxy: mean proj %.2f sd %.2f min %.2f max %.2f'%(p.mean(),p.std(),p.min(),p.max()))
np.save('work/014_mu23.npy',mu.numpy())
