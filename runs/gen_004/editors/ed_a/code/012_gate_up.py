# Hypothesis: n5407's strong negative activation at commit points comes from an open gate (w1.x>0) times a negative up (w3.x<0); reconstruct gate/up from recorded h19 minus the recorded MLP18 output, check reconstruction, and measure how the conclusion direction projects on the L18 MLP input.
import json, numpy as np, torch
from wsp.editor import *
items=json.load(open('work/read1_items.json'))
a=torch.tensor(np.load('work/fwd_01/mlp18.npy').astype(np.float32))
h19=torch.tensor(np.load('work/fwd_01/h19.npy').astype(np.float32))
W2=load('model.layers.18.feed_forward.w2.weight'); gp=load('model.layers.18.post_ffn_norm.weight'); gn=load('model.layers.18.ffn_norm.weight')
o=a@W2.T; y=gp*o/o.pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
x=h19-y; xh=gn*x/x.pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
w1=load('model.layers.18.feed_forward.w1.weight')[5407]; w3=load('model.layers.18.feed_forward.w3.weight')[5407]
g=xh@w1; u=xh@w3; rec=torch.nn.functional.silu(g)*u
print('reconstruction corr %.4f, mean abs err %.3f'%(np.corrcoef(rec,a[:,5407])[0,1], float((rec-a[:,5407]).abs().mean())))
np.save('work/l18_xhat.npy',xh.numpy())
G={}
for j,it in enumerate(items):
    k=(it['kind'],it['label'] if it['kind']=='concl' else 'ctrl',it['cut']); G.setdefault(k,[]).append(j)
for k,js in sorted(G.items()):
    print(k,len(js),'gate %.2f  up %.2f  act %.2f  frac(gate>0) %.2f'%(g[js].mean(),u[js].mean(),a[js,5407].mean(),(g[js]>0).float().mean()))
