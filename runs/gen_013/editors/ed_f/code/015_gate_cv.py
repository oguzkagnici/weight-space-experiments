# Hypothesis: at the L23 MLP input a cross-validated error detector reaches a usable operating point (TPR at 5% FPR on held-out clean steps and controls).
import json, numpy as np
from sklearn.metrics import roc_auc_score
D='work/fwd_02'; M=json.load(open('work/p2_meta.json')); N=len(M)
for name in ('in_model.layers.23.feed_forward','in_model.layers.22.feed_forward','h22','h24'):
    X=np.load(D+'/%s.npy'%name).astype(np.float64)
    src=np.array([m['src'] for m in M]); kind=np.array([m.get('kind','') for m in M]); fam=np.array([m.get('fam','') for m in M]); qid=np.array([m.get('id','') for m in M])
    cl=np.where((src=='p1')&(kind=='clean'))[0]; co=np.where((src=='p1')&(kind=='big'))[0]; ctl=np.where((src=='ctlA')|(src=='ctlB'))[0]
    qs=np.array(sorted(set(qid))); rng=np.random.RandomState(1); rng.shuffle(qs); folds=np.array_split(qs,6)
    for lam in (0.3,3,30,1e4):
        sc=np.full(N,np.nan)
        for fo in folds:
            te_q=set(fo); tr=np.array([q not in te_q for q in qid[cl]]); ctr=ctl[[qid[i] not in te_q for i in ctl]]
            Dm=X[co][tr]-X[cl][tr]; d=Dm.mean(0)
            Z=np.vstack([X[cl][tr]-X[cl][tr].mean(0),X[ctr]-X[ctr].mean(0),Dm-d]); n=len(Z); s=np.trace(Z@Z.T)/n/X.shape[1]
            w=(d-Z.T@np.linalg.solve(Z@Z.T/n+lam*s*np.eye(n),Z@d)/n)/(lam*s); w/=np.linalg.norm(w)
            te=np.array([q in te_q for q in qid]); sc[te]=X[te]@w
        ok=~np.isnan(sc)
        neg=np.r_[cl,ctl]; neg=neg[ok[neg]]
        out=[]
        for fpr in (.05,.10):
            th=np.quantile(sc[neg],1-fpr); out.append('FPR %.2f: TPR F1 %.2f F2 %.2f (clean %.2f, ctl %.2f)'%(fpr,(sc[co[fam[co]=='F1']]>th).mean(),(sc[co[fam[co]=='F2']]>th).mean(),(sc[cl]>th).mean(),(sc[ctl[ok[ctl]]]>th).mean()))
        a=lambda f:roc_auc_score(np.r_[np.zeros((fam[cl]==f).sum()),np.ones((fam[co]==f).sum())],np.r_[sc[cl[fam[cl]==f]],sc[co[fam[co]==f]]])
        print('%-34s lam %-7g AUC F1 %.3f F2 %.3f | corrupt vs controls %.3f | '%(name,lam,a('F1'),a('F2'),roc_auc_score(np.r_[np.zeros(ok[ctl].sum()),np.ones(len(co))],np.r_[sc[ctl[ok[ctl]]],sc[co]]))+' ; '.join(out))
