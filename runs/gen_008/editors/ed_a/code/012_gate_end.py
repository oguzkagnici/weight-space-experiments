# Hypothesis: a shrinkage-LDA direction at the L23 MLP input detects sweep-end breaks (grouped-CV AUC high) with a tiny all-token firing rate;
# also: what does the parent write at sweep-end breaks (top next tokens, P(Hence/Thus/Therefore))?
import json, numpy as np, sys, collections
from sklearn.metrics import roc_auc_score
from wsp.editor import tok
fwd='work/fwd_01'; layer=sys.argv[1] if len(sys.argv)>1 else '23'
X=np.load(f"{fwd}/in_model.layers.{layer}.feed_forward.w1.npy", mmap_mode="r")
R=np.load('work/labels_end_p1.npy'); ti=R[:,0].astype(int); isb=R[:,1]==1; pos=R[:,2]==1
rng=np.random.default_rng(0); allidx=np.sort(rng.choice(len(R),40000,replace=False))
brk=np.where(isb)[0]
Xa=np.asarray(X[allidx],dtype=np.float32); Xb=np.asarray(X[brk],dtype=np.float32)
mu=Xa.mean(0); C=np.cov(Xa.T); lam=0.1*np.trace(C)/C.shape[0]; Ci=np.linalg.inv(C+lam*np.eye(C.shape[0]))
yb=pos[brk]; grp=ti[brk]
def fitdir(m): return Ci@(Xb[m&yb].mean(0)-Xb[m&~yb].mean(0))
texts=np.unique(ti); rng.shuffle(texts); folds=np.array_split(texts,5); sc=np.zeros(len(brk))
for f in folds:
    te=np.isin(grp,f); sc[te]=Xb[te]@fitdir(~te)
print('L',layer,'grouped-CV AUC sweep-end vs other breaks %.3f'%roc_auc_score(yb,sc))
d=fitdir(np.ones(len(brk),bool)); sa=Xa@d; m,s=sa.mean(),sa.std(); zb=(Xb@d-m)/s; za=(sa-m)/s
zcv=(sc-sc[~yb].mean())/sa.std()  # rough
print('in-sample z: pos %.2f other breaks %.2f  all 99.9pct %.2f max %.2f'%(zb[yb].mean(),zb[~yb].mean(),np.percentile(za,99.9),za.max()))
for thr in [5,6,7,8,9,10]:
    print(' thr',thr,'recall %.2f'%(zb[yb]>thr).mean(),'false breaks',int((zb[~yb]>thr).sum()),'all-token rate %.5f'%(za>thr).mean(), 'CV recall@same-FP',
          '%.2f'%(sc[yb]>np.sort(sc[~yb])[-max(1,int((zb[~yb]>thr).sum()))]).mean())
np.save(f'work/end_d_L{layer}.npy',d); np.save(f'work/end_stats_L{layer}.npy',np.array([m,s])); np.save(f'work/mu_L{layer}.npy',mu); np.save(f'work/Ci_L{layer}.npy',Ci.astype(np.float32))
if layer=='23':
    I=np.load(f'{fwd}/topk_ids.npy',mmap_mode='r'); V=np.load(f'{fwd}/topk_logprobs.npy',mmap_mode='r')
    P=collections.Counter(); Pc=collections.Counter(); n=0
    for r in np.where(pos)[0]:
        n+=1
        for i,v in zip(I[r],V[r]): P[tok.decode([int(i)])]+=float(np.exp(v))
    print('mean next-token probs at sweep-end breaks (n=%d):'%n, [(w,round(p/n,3)) for w,p in P.most_common(25)])
    for w in ['Hence','Thus','Therefore','So','We','Given']:
        print(w, round(P[w]/n,3))
