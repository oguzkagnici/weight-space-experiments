# Hypothesis: a whitened (shrinkage-LDA) direction, or single MLP neurons, separate corrupted from clean steps much better than the mean difference (held-out by question).
import json, sys, numpy as np
from sklearn.metrics import roc_auc_score
D='work/fwd_01'; meta=json.load(open('work/p1_meta.json')); N=len(meta)
fam=np.array([m['fam'] for m in meta]); kind=np.array([m['kind'] for m in meta]); qid=np.array([m['id'] for m in meta])
grp=lambda f,k:np.where((fam==f)&(kind==k))[0]
a=np.r_[grp('F1','clean'),grp('F1','clean'),grp('F2','clean')]; b=np.r_[grp('F1','big'),grp('F1','small'),grp('F2','big')]
tag=np.array(['F1big']*80+['F1small']*80+['F2']*33)
qs=np.array(sorted(set(qid))); rng=np.random.RandomState(0); rng.shuffle(qs); folds=np.array_split(qs,6)
def cv(X,lam):
    sa=np.zeros(len(a)); sb=np.zeros(len(a))
    for fo in folds:
        te=np.isin(qid[a],fo); tr=~te
        Dm=X[b][tr]-X[a][tr]; d=Dm.mean(0)
        # within covariance: clean states of training questions (centred) + paired-difference residuals
        C=X[np.unique(a[tr])]; C=C-C.mean(0); R=Dm-d
        Z=np.vstack([C,R]); 
        # dual-form ridge: w = (Z^T Z/n + lam I)^-1 d
        n=len(Z); s=np.trace(Z@Z.T)/n/X.shape[1]
        K=Z@Z.T/n; w=(d-Z.T@np.linalg.solve(K+lam*s*np.eye(n),Z@d)/n)/(lam*s)
        sa[te]=X[a][te]@w; sb[te]=X[b][te]@w
    out=[]
    for t in ('F1big','F1small','F2'):
        m=tag==t; out.append((roc_auc_score(np.r_[np.zeros(m.sum()),np.ones(m.sum())],np.r_[sa[m],sb[m]]),(sb[m]>sa[m]).mean()))
    return out
for L in (14,18,20,22,23,24):
    X=np.load(D+'/h%d.npy'%L).astype(np.float64)
    for lam in (0.03,0.3,3):
        r=cv(X,lam); print('h%-2d lam %.2f | '%(L,lam)+' | '.join('%s AUC %.3f pair %.2f'%(t,x,y) for t,(x,y) in zip(('F1big','F1small','F2'),r)))
# single neurons
for L in (20,21,22,23):
    H=np.load(D+'/mlp%d.npy'%L).astype(np.float64)
    half=np.isin(qid[a],np.concatenate(folds[:3]))
    dtr=(H[b][half]-H[a][half]); t=dtr.mean(0)/(dtr.std(0)+1e-6)*np.sqrt(half.sum())
    top=np.argsort(-np.abs(t))[:8]
    te=~half
    print('mlp%d top neurons by paired t (train half) -> test-half AUC/pair:'%L)
    for j in top:
        s=np.sign(t[j]); au=roc_auc_score(np.r_[np.zeros(te.sum()),np.ones(te.sum())],s*np.r_[H[a][te][:,j],H[b][te][:,j]])
        print('   n%-5d t %+.1f | AUC %.3f pair %.2f | mean clean %.3f corrupt %.3f | absmax all %.2f'%(j,t[j],au,(s*(H[b][te][:,j]-H[a][te][:,j])>0).mean(),H[a][:,j].mean(),H[b][:,j].mean(),np.abs(H[:,j]).max()))
