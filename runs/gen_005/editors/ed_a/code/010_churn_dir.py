# Hypothesis: a mean-difference direction at the MLP input (L20, L23) at paragraph breaks separates breaks followed by a churn opener from other breaks (cross-validated by trace), and scores low at finished-trace breaks and at ordinary tokens.
import json, numpy as np
from sklearn.metrics import roc_auc_score
from wsp.editor import *
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json'))
topi=np.load(f'{D}/topk_ids.npy'); topv=np.load(f'{D}/topk_logprobs.npy')
CH=[tok.encode(s).ids[0] if hasattr(tok.encode(s),'ids') else None for s in []]
churn_ids=[91780]+[tok(s,add_special_tokens=False).input_ids[0] for s in ['Maybe','Perhaps','Another','Possibly']]
commit_ids=[tok(s,add_special_tokens=False).input_ids[0] for s in ['Thus','Hence','Therefore']]
print('churn ids',churn_ids,'commit ids',commit_ids)
rowsI=np.array([r['row'] for r in recs])
P=np.exp(topv[rowsI]); I=topi[rowsI]
pch=(P*np.isin(I,churn_ids)).sum(1); pcm=(P*np.isin(I,commit_ids)).sum(1)
cut=np.array([r['cut'] for r in recs]); cls=np.array([r['cls'] for r in recs]); ti=np.array([r['ti'] for r in recs]); frac=np.array([r['frac'] for r in recs])
for c in (0,1):
    m=cut==c
    print('cut',c,'P(churn opener) mean %.3f  P(Thus/Hence/Therefore) %.3f'%(pch[m].mean(),pcm[m].mean()),
          ' at churn breaks %.3f'%(pch[m&(cls=='churn')].mean() if (m&(cls=='churn')).any() else -1),' at other %.3f'%pch[m&(cls=='other')].mean())
for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.01)):
    print('frac %.2f-%.2f'%(lo,hi),' cut pch %.3f'%pch[(cut==1)&(frac>=lo)&(frac<hi)].mean(),' fin pch %.3f'%pch[(cut==0)&(frac>=lo)&(frac<hi)].mean())
print('AUC of model P(churn) for churn vs other breaks: %.3f'%roc_auc_score(cls[cls!='commit']=='churn',pch[cls!='commit']))
print('AUC of model P(churn) for cut vs finished breaks: %.3f'%roc_auc_score(cut,pch))
np.save('work/010_pch.npy',pch)
for L in (20,23):
    X=np.load(f'{D}/in_model.layers.{L}.feed_forward.w1.npy',mmap_mode='r')
    Xb=np.asarray(X[rowsI],dtype=np.float32)
    # all-token baseline sample
    rng=np.random.default_rng(0); samp=rng.choice(X.shape[0],20000,replace=False); samp.sort()
    Xs=np.asarray(X[samp],dtype=np.float32)
    # 4-fold CV by trace, label = churn vs other (all traces)
    folds=ti%4; sc=np.zeros(len(recs))
    lab=(cls=='churn')
    for f in range(4):
        tr=(folds!=f)&(cls!='commit')
        d=Xb[tr&lab].mean(0)-Xb[tr&~lab].mean(0)
        sc[folds==f]=Xb[folds==f]@d/np.linalg.norm(d)
    m=cls!='commit'
    print(f'L{L}: CV AUC churn vs other breaks %.3f'%roc_auc_score(lab[m],sc[m]))
    d=Xb[lab].mean(0)-Xb[(~lab)&(cls!='commit')].mean(0); dn=d/np.linalg.norm(d)
    s=Xb@dn; ss=Xs@dn
    print('   full-dir score: churn %.2f other-cut %.2f other-fin %.2f commit %.2f all-tokens mean %.2f sd %.2f p99.5 %.2f'%(s[lab].mean(),s[(cls=='other')&(cut==1)].mean(),s[(cls=='other')&(cut==0)].mean(),s[cls=='commit'].mean(),ss.mean(),ss.std(),np.percentile(ss,99.5)))
    print('   corr(score, model P(churn)) at breaks %.3f'%np.corrcoef(s,pch)[0,1])
    np.save(f'work/010_d{L}.npy',d)
