# Hypothesis: the large negative activations of v3 occur only at prompt positions (BOS-like tokens with negative bias-proxy projection), which cannot affect generation because L23's output only feeds the logits at its own position.
import json, numpy as np, torch
from wsp.editor import *
D='work/fwd_01'
X=np.load(f'{D}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); offs=np.load(f'{D}/offsets.npy')
ids_list=json.load(open('work/req1_ids.json'))
W1=load('model.layers.23.feed_forward.w1.weight')[74]; W3=load('model.layers.23.feed_forward.w3.weight')[74]
S=json.load(open('work/026_params.json'))['S']
gen=np.zeros(X.shape[0],bool)
for k,q in enumerate(ids_list):
    pl=len(tok(render_think(q),add_special_tokens=False).input_ids); gen[offs[k]+pl-1:offs[k+1]]=True
acts=np.concatenate([(torch.nn.functional.silu(torch.tensor(np.asarray(X[i:i+20000],np.float32))@W1)*(torch.tensor(np.asarray(X[i:i+20000],np.float32))@W3)).numpy() for i in range(0,X.shape[0],20000)])
np.save('work/027_act_all.npy',acts.astype(np.float32))
print('generated positions:',gen.sum(),' min act %.3e  n(act<-0.01*S) %d  n(act>0.1*S) %d'%(acts[gen].min(),(acts[gen]<-0.01*S).sum(),(acts[gen]>0.1*S).sum()))
print('prompt positions: min act %.3f n(act<-0.01*S) %d'%(acts[~gen].min(),(acts[~gen]<-0.01*S).sum()))
