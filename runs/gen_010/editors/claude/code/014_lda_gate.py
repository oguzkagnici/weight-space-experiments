# Hypothesis: a shrinkage-LDA direction (hedged commitment breaks vs firm breaks + other tokens) is specific enough to gate
# a unit: grouped (leave-one-text-out) estimate of the share of other tokens / firm breaks above a threshold that keeps 50%
# of hedged breaks.
import json, numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
items=json.load(open('work/read4_index.json')); meta=json.load(open('work/fwd_04/meta.json'))
X=np.load('work/fwd_04/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
offs=np.cumsum([0]+meta['n_tokens']); rng=np.random.default_rng(0)
rows=[];lab=[];grp=[]
for t,(it,o) in enumerate(zip(items,offs[:-1])):
    mk={ti:h for ti,h,_ in it['marks']}
    for ti,h in mk.items(): rows.append(o+ti); lab.append(2 if h else 1); grp.append(t)
    gen=np.arange(o+it['plen'],o+it['n']); gen=gen[~np.isin(gen,[o+k for k in mk])]
    for p in rng.choice(gen,size=min(400,len(gen)),replace=False): rows.append(p); lab.append(0); grp.append(t)
A=np.asarray(X[np.array(rows)],dtype=np.float32); lab=np.array(lab); grp=np.array(grp)
folds=np.array_split(np.random.default_rng(1).permutation(len(items)),8)
sc=np.zeros(len(A))
for f in folds:
    te=np.isin(grp,f); m=LDA(solver='lsqr',shrinkage='auto').fit(A[~te],(lab[~te]==2).astype(int))
    sc[te]=m.decision_function(A[te])
th=np.percentile(sc[lab==2],50)
print('held-out folds: hedged above th 50%; firm above:',round(float((sc[lab==1]>th).mean()),3),'other above:',round(float((sc[lab==0]>th).mean()),4),
      '| th at 30% hedged: other', round(float((sc[lab==0]>np.percentile(sc[lab==2],70)).mean()),4))
from sklearn.metrics import roc_auc_score
print('AUC hedged vs firm',round(roc_auc_score(lab[lab>0]==2,sc[lab>0]),3),'hedged vs other',round(roc_auc_score(lab[lab!=1]==2,sc[lab!=1]),4))
