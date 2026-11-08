# Hypothesis: at the L23 MLP input, (A) eval sweep-end breaks separate from other breaks/tokens and (B) a forced paragraph-initial "Hence" after a sweep end separates from all other tokens, by shrinkage-LDA; a min-variance bias proxy has near-constant projection.
import json, numpy as np, glob
from sklearn.metrics import roc_auc_score
rows=json.load(open('work/p1_rows.json'))
f=[p for p in glob.glob('work/fwd_01/*.npy') if 'w1' in p][0]
X=np.load(f,mmap_mode='r'); print(f,X.shape)
rng=np.random.default_rng(0)
lab=np.array([r[3] for r in rows]); gid=np.array([r[0] for r in rows]); qid=np.array([r[2] for r in rows])
def get(mask): return np.asarray(X[np.sort(gid[mask])],dtype=np.float64), qid[mask][np.argsort(gid[mask])]
think=np.array([not l.startswith('prompt') and not l.startswith('forced') and not l.endswith('_dup') for l in lab])
allidx=np.nonzero(think)[0]; samp=rng.choice(allidx,30000,replace=False)
Xa,_=get(np.isin(np.arange(len(rows)),samp))
mu=Xa.mean(0); C=np.cov(Xa.T)
lam=0.05*np.trace(C)/C.shape[0]
Ci=np.linalg.inv(C+lam*np.eye(2048))
p=Ci@mu/(mu@Ci@mu)
Xp,_=get(np.array([l.startswith('prompt') for l in lab]))
proj=Xa@p; print('bias proxy: mean %.3f sd %.3f min %.3f max %.3f | prompt min %.3f max %.3f'%(proj.mean(),proj.std(),proj.min(),proj.max(),(Xp@p).min(),(Xp@p).max()))
np.save('work/proxy.npy',p)
def lda(Xp_,Xn_,Cs):
    d=np.linalg.solve(Cs+lam*np.eye(2048),Xp_.mean(0)-Xn_.mean(0)); return d
def fit(pos_mask,neg_mask,name):
    Xpos,qp=get(pos_mask); Xneg,qn=get(neg_mask)
    print(name,'pos',len(Xpos),'neg',len(Xneg))
    # grouped CV
    uq=np.unique(np.concatenate([qp,qn])); rng.shuffle(uq); folds=np.array_split(uq,5)
    sc_p=np.zeros(len(Xpos)); sc_n=np.zeros(len(Xneg))
    for fo in folds:
        tp=~np.isin(qp,fo); tn=~np.isin(qn,fo)
        Cs=np.cov(np.vstack([Xpos[tp]-Xpos[tp].mean(0),Xneg[tn]-Xneg[tn].mean(0)]).T)*0.5+C*0.5
        d=lda(Xpos[tp],Xneg[tn],Cs)
        sc_p[~tp]=Xpos[~tp]@d; sc_n[~tn]=Xneg[~tn]@d
    print('  grouped-CV AUC %.3f'%roc_auc_score(np.r_[np.ones(len(sc_p)),np.zeros(len(sc_n))],np.r_[sc_p,sc_n]))
    Cs=np.cov(np.vstack([Xpos-Xpos.mean(0),Xneg-Xneg.mean(0)]).T)*0.5+C*0.5
    d=lda(Xpos,Xneg,Cs)
    s_all=Xa@d; m0=np.median(s_all); sd=s_all.std()
    z=lambda Y:(Y@d-m0*(Y@p))/sd
    za=z(Xa); zp=z(Xpos); zn=z(Xneg); zpr=z(Xp)
    print('  z all: p99 %.1f p99.9 %.1f p99.99 %.1f max %.1f | prompt max %.1f'%tuple(np.percentile(za,[99,99.9,99.99]).tolist()+[za.max(),zpr.max()]))
    print('  z pos: pct10/50/90 %.1f %.1f %.1f | neg p50/p90/p99/max %.1f %.1f %.1f %.1f'%tuple(np.percentile(zp,[10,50,90]).tolist()+np.percentile(zn,[50,90,99]).tolist()+[zn.max()]))
    np.save(f'work/dir_{name}.npy',np.r_[d/sd,[m0/sd]])
    return d,m0,sd
A=fit(lab=='swend',np.isin(lab,['brk','lisend']),'A')
B=fit(lab=='forced1',(np.isin(np.arange(len(rows)),samp)|np.isin(lab,['brk','swend','lisend','hence_init'])),'B')
# B: how do natural paragraph-initial Hence score
d,m0,sd=B; Xh,_=get(lab=='hence_init'); print('natural para-initial Hence n',len(Xh),'zB', np.round((Xh@d-m0*(Xh@p))/sd,1))
d,m0,sd=A; Xh,_=get(lab=='forced1'); print('zA at forced Hence', np.percentile((Xh@d-m0*(Xh@p))/sd,[10,50,90]).round(1))
