# Hypothesis: some late-layer neurons are effectively dead along the traces (max |act| * ||g_post*w2col|| << n74's 0.033); repurposing one would make the edit invisible at non-firing positions and remove the butterfly divergences.
import numpy as np, torch
from wsp.editor import *
D='work/fwd_01'
X=np.load(f'{D}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
a_all=None
L=23
w1=load_original(f'model.layers.{L}.feed_forward.w1.weight'); w3=load_original(f'model.layers.{L}.feed_forward.w3.weight'); w2=load_original(f'model.layers.{L}.feed_forward.w2.weight'); gp=load(f'model.layers.{L}.post_ffn_norm.weight')
cn=(w2*gp[:,None]).norm(dim=0)
mx=torch.zeros(7680); mean=torch.zeros(7680); n=0
gen_mask=None
for i in range(0,X.shape[0],8000):
    Xc=torch.tensor(np.asarray(X[i:i+8000],np.float32))
    A=(torch.nn.functional.silu(Xc@w1.T)*(Xc@w3.T)).abs()
    mx=torch.maximum(mx,A.max(0).values); mean+=A.sum(0); n+=A.shape[0]
mean/=n
c_max=mx*cn; c_mean=mean*cn
o=torch.argsort(c_max)[:10]
print('n74: max contrib %.4f mean %.5f'%(c_max[74],c_mean[74]))
for i in o: print(int(i),'max|a| %.4f mean|a| %.5f  max contrib %.5f mean contrib %.6f'%(mx[i],mean[i],c_max[i],c_mean[i]))
