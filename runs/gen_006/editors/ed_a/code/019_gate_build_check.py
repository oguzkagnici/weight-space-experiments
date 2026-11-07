# Hypothesis: gate G = d.x - t*(b.x)/mean_b with PC-cleaned d (target mean-diff) and PC-cleaned mean proxy b stays below 0 on
# held-out thinking tokens and above 0 on held-out targets (p0 and p6) when t sits between thinking/content and target scores.
import json, numpy as np
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json')); n=json.load(open(D+'meta.json'))['n_tokens']
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
def seg(js): return np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in js])
def nat(js): return np.array([off[j]+cpos[j]+k for j in js for k in range(1,60) if cpos[j]+k<n[j]])
def build(train, tgt_rows, zstar):
    Xk=np.asarray(X[seg(train)],dtype=np.float32); mu=Xk.mean(0)
    ev,U=np.linalg.eigh(np.cov((Xk-mu).T)); P=U[:,::-1][:,:200]
    cl=lambda v:v-P@(P.T@v)
    b=cl(mu); b/=np.linalg.norm(b); mb=float((Xk@b).mean())
    Tk=np.asarray(X[tgt_rows],dtype=np.float32); d=cl(Tk.mean(0)-mu)
    s=Xk@d; t=s.mean()+zstar*s.std()
    w=d-(t/mb)*b
    return w, s.std()
for k,zstar in ((0,15.0),(6,23.0)):
    for tr,te in (([0,2,4,6,8,10,12],[1,3,5,7,9,11]),([1,3,5,7,9,11],[0,2,4,6,8,10,12])):
        ttr=[off[j]+cpos[j]+k for j in range(13,26) if (j-13) in tr]; tte=[off[j]+cpos[j]+k for j in range(13,26) if (j-13) in te]
        w,sd=build(tr,ttr,zstar)
        g=lambda rows:(np.asarray(X[rows],dtype=np.float32)@w)/sd
        gk=g(seg(te)); gt=g(tte); gn=g(nat(te)); gtr=g(ttr)
        print(f'p{k} z*={zstar}: held-out thinking max {gk.max():.1f} (p99.99 {np.percentile(gk,99.99):.1f}, mean {gk.mean():.1f}) | held-out natural content max {gn.max():.1f} | held-out targets min {gt.min():.1f} median {np.median(gt):.1f} | train targets min {gtr.min():.1f}')
