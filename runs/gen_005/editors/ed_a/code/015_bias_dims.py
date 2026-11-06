# Hypothesis: the L23 MLP input has a few "massive" dimensions with near-constant value that make a better bias proxy than the mean direction.
import json, numpy as np
D='work/fwd_01'
X=np.load(f'{D}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
rng=np.random.default_rng(2); samp=np.sort(rng.choice(X.shape[0],40000,replace=False))
Xs=np.asarray(X[samp],np.float32)
recs=json.load(open('work/009_breaks.json')); Xb=np.asarray(X[np.array([r['row'] for r in recs])],np.float32)
m=Xs.mean(0); s=Xs.std(0); r=np.abs(m)/s
o=np.argsort(-r)[:8]
for i in o: print(i,'mean %.2f sd %.2f  breaks mean %.2f sd %.2f'%(m[i],s[i],Xb[:,i].mean(),Xb[:,i].std()))
# combined: sum of sign-weighted top dims
k=o[:5]; w=np.zeros(2048); w[k]=np.sign(m[k])/s[k]
p=Xs@w; pb=Xb@w
print('combo top5: mean %.2f sd %.2f (cv %.3f) | breaks mean %.2f sd %.2f'%(p.mean(),p.std(),p.std()/p.mean(),pb.mean(),pb.std()))
np.save('work/015_biasw.npy',w)
