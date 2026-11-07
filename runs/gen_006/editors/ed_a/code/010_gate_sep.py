# Hypothesis: the L23 MLP input at </think> (p0) and the first content tokens (p1..p6) is linearly separable from every
# prompt+thinking token by a mean-difference direction with a wide margin.
import json, numpy as np
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json'))
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
think=[]; 
for j in range(13):
    idx=np.arange(off[j],off[j]+cpos[j]); think.append(idx)
think=np.concatenate(think)
rng=np.random.default_rng(0)
Xt=np.asarray(X[think],dtype=np.float32)
mu_t=Xt.mean(0)
res={}
for k in range(0,7):
    pos=np.array([off[j]+cpos[j]+k for j in range(13,26)])   # variants: </think>,The,answer,is,(,X,).
    Xp=np.asarray(X[pos],dtype=np.float32)
    # leave-one-out: direction from 12 targets, score held-out one
    loo=[]
    for i in range(len(pos)):
        m=np.delete(Xp,i,0).mean(0)-mu_t; d=m/np.linalg.norm(m)
        loo.append(Xp[i]@d - 0)
    d=(Xp.mean(0)-mu_t); d/=np.linalg.norm(d)
    st=Xt@d; sp=Xp@d
    res[k]=d
    print(f'p{k}: thinking scores max {st.max():.2f} p99.99 {np.percentile(st,99.99):.2f} mean {st.mean():.2f} sd {st.std():.2f} | target min {sp.min():.2f} mean {sp.mean():.2f} | LOO target min {min(loo):.2f}')
np.save('work/p1_dirs23.npy',np.stack([res[k] for k in range(7)]))
np.save('work/p1_mu_think23.npy',mu_t)
# bias proxy: unit mean direction of thinking tokens
b=mu_t/np.linalg.norm(mu_t); pb=Xt@b
print('bias proxy projection: mean %.3f sd %.3f min %.3f max %.3f'%(pb.mean(),pb.std(),pb.min(),pb.max()))
print('norm of x-hat: mean %.2f'%np.linalg.norm(Xt[:2000],axis=1).mean())
