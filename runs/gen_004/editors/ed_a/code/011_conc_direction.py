# Hypothesis: the residual stream at a paragraph break carries a linearly decodable "a conclusion (letter) was just stated" feature, separable from control breaks, that could gate the commit neuron; measure CV AUC of mean-difference directions per layer (concl vs ctrl; commit vs reconsider), split by question.
import json, numpy as np
from sklearn.metrics import roc_auc_score
items=json.load(open('work/read1_items.json'))
qids=sorted(set(it['id'] for it in items)); rng=np.random.default_rng(0); fold={q:rng.integers(2) for q in qids}
f=np.array([fold[it['id']] for it in items])
isc=np.array([it['kind']=='concl' for it in items]); iscom=np.array([it['label']=='commit' for it in items])
for l in [10,12,14,16,17,18,19,20,22]:
    h=np.load(f'work/fwd_01/h{l}.npy').astype(np.float32); h=h/np.linalg.norm(h,axis=1,keepdims=True)
    out=[]
    for name,y,mask in [('concl-vs-ctrl',isc,np.ones(len(items),bool)),('commit-vs-rec',iscom,isc)]:
        sc=np.zeros(len(items))
        for k in [0,1]:
            tr=(f!=k)&mask; d=h[tr&y].mean(0)-h[tr&~y].mean(0); sc[f==k]=h[f==k]@d
        out.append('%s AUC %.2f'%(name,roc_auc_score(y[mask],sc[mask])))
    print('h%d'%l,' | '.join(out))
