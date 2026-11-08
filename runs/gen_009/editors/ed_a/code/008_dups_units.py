# Hypothesis: L23 duplicate groups {331,5105,881} and {5821,2400} are still intact in the parent; the parent's L23 units show usable scales.
import torch, numpy as np
from wsp.editor import *
L=23
w1=load(f'model.layers.{L}.feed_forward.w1.weight'); w3=load(f'model.layers.{L}.feed_forward.w3.weight'); w2=load(f'model.layers.{L}.feed_forward.w2.weight')
b1=load_base(f'model.layers.{L}.feed_forward.w1.weight')
print(w1.shape,w2.shape)
def cos(a,b): return float(a@b/(a.norm()*b.norm()))
for g in [(331,5105,881),(5821,2400),(1228,3401,3793,6986)]:
    k=g[0]
    for j in g[1:]:
        print(g,j,'cos w1 %.5f w3 %.5f w2 %.5f'%(cos(w1[j],w1[k]),cos(w3[j],w3[k]),cos(w2[:,j],w2[:,k])),
              'norm ratio w1 %.4f w3 %.4f w2 %.4f'%(w1[j].norm()/w1[k].norm(),(w3[j]@w3[k])/(w3[k]@w3[k]),w2[:,j].norm()/w2[:,k].norm()),
              'changed vs base', bool((w1[j]-b1[j]).abs().max()>0))
# parent units
for j in [3401,3793,6986,1228]:
    print('unit',j,'|w1| %.2f |w3| %.2f |w2| %.3f max|w2| %.4f'%(w1[j].norm(),w3[j].norm(),w2[:,j].norm(),w2[:,j].abs().max()))
print('typical |w1| %.3f |w3| %.3f |w2col| %.3f'%(w1.norm(dim=1).median(),w3.norm(dim=1).median(),w2.norm(dim=0).median()))
# search other near-duplicates via w1 cos
W=w1/w1.norm(dim=1,keepdim=True)
G=W@W.T; G.fill_diagonal_(0)
v,i=G.abs().max(dim=1)
idx=torch.nonzero(v>0.99).flatten().tolist()
print('w1 |cos|>0.99 neurons:',[(a,int(i[a]),round(float(G[a,i[a]]),4)) for a in idx])
pf=load('model.layers.23.post_ffn_norm.weight'); fn=load('model.embedding_norm.weight'); ff=load('model.layers.23.ffn_norm.weight')
print('post_ffn gain mean %.3f final norm gain mean %.3f ffn_norm mean %.3f'%(pf.abs().mean(),fn.abs().mean(),ff.abs().mean()))
