# Hypothesis: at the L23 (and L22) MLP input, a shrinkage-LDA direction separates paragraph breaks that start a 2nd/3rd option
# sweep from all other breaks (incl. 1st-sweep starts), with grouped-CV AUC well above chance and near-zero firing elsewhere.
import json, numpy as np, sys, glob, os
from sklearn.metrics import roc_auc_score
fwd=sys.argv[1]; layer=sys.argv[2] if len(sys.argv)>2 else '23'
meta=json.load(open(f'{fwd}/meta.json'))
fn=[f for f in meta['files'] if f'layers.{layer}.feed_forward.w1' in f][0]
X=np.load(f"{fwd}/in_model.layers.{layer}.feed_forward.w1.npy", mmap_mode="r")
R=np.load('work/labels_p1.npy'); assert len(R)==X.shape[0], (len(R), X.shape)
ti=R[:,0].astype(int); isb=R[:,1]==1; pos=R[:,2]==1; first=R[:,3]==1; near=R[:,4]==1
rng=np.random.default_rng(0)
allidx=rng.choice(len(R), 40000, replace=False)
brk=np.where(isb)[0]
Xa=np.asarray(X[allidx],dtype=np.float32); Xb=np.asarray(X[brk],dtype=np.float32)
mu=Xa.mean(0); C=np.cov(Xa.T)
lam=0.1*np.trace(C)/C.shape[0]
Ci=np.linalg.inv(C+lam*np.eye(C.shape[0]))
yb=pos[brk]; grp=ti[brk]
negmask=~yb
def fitdir(mask_tr):
    mp=Xb[mask_tr&yb].mean(0); mn=Xb[mask_tr&negmask].mean(0)
    d=Ci@(mp-mn); return d
# grouped 5-fold CV by text
texts=np.unique(ti); rng.shuffle(texts); folds=np.array_split(texts,5)
sc=np.zeros(len(brk))
for f in folds:
    te=np.isin(grp,f); d=fitdir(~te); sc[te]=Xb[te]@d
print('L',layer,'CV AUC pos vs all other breaks %.3f'%roc_auc_score(yb,sc),
      ' pos vs first-sweep breaks %.3f'%roc_auc_score(yb[yb|first[brk]], sc[yb|first[brk]]))
d=fitdir(np.ones(len(brk),bool))
sa=Xa@d; sbk=Xb@d
m,s=sa.mean(),sa.std(); z_b=(sbk-m)/s; z_a=(sa-m)/s
print('z: pos mean %.2f  first mean %.2f  other breaks mean %.2f  all-token pctl 99.9 %.2f max %.2f'%(z_b[yb].mean(), z_b[first[brk]].mean(), z_b[negmask&~first[brk]].mean(), np.percentile(z_a,99.9), z_a.max()))
for thr in [4,5,6,7,8]:
    fp=(z_b[negmask]>thr); print(' thr',thr,'pos recall %.2f'%(z_b[yb]>thr).mean(),'false breaks',int(fp.sum()),'(near-sweep',int((fp&near[brk][negmask]).sum()),') all-token rate %.5f'%(z_a>thr).mean(),
          'pos cut traces hit',len(set(grp[yb&(z_b>thr)&(R[brk,7]==1)])), 'fin traces w/ fire',len(set(grp[(z_b>thr)&(R[brk,7]==0)])))
np.save(f'work/gate_d_L{layer}.npy',d); np.save(f'work/gate_stats_L{layer}.npy',np.array([m,s]))
np.save(f'work/mu_L{layer}.npy',mu); np.save(f'work/Ci_L{layer}.npy',Ci.astype(np.float32))
