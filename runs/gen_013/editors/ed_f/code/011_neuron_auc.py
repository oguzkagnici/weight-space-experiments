# Hypothesis: single late-MLP neurons (e.g. L22 n4847, x2 after a corrupted step) detect the corrupted step with usable unpaired AUC.
import json, numpy as np
from sklearn.metrics import roc_auc_score
D='work/fwd_01'; meta=json.load(open('work/p1_meta.json'))
fam=np.array([m['fam'] for m in meta]); kind=np.array([m['kind'] for m in meta])
grp=lambda f,k:np.where((fam==f)&(kind==k))[0]
sets={'F1big':(grp('F1','clean'),grp('F1','big')),'F1small':(grp('F1','clean'),grp('F1','small')),'F2':(grp('F2','clean'),grp('F2','big'))}
for L in (18,19,20,21,22,23):
    H=np.load(D+'/mlp%d.npy'%L).astype(np.float32)
    a,b=sets['F1big']; 
    # rank-based AUC for all neurons on F1big
    from scipy.stats import rankdata
    R=rankdata(np.vstack([H[a],H[b]]),axis=0); auc=(R[len(a):].sum(0)-len(b)*(len(b)+1)/2)/(len(a)*len(b))
    top=np.argsort(-np.abs(auc-.5))[:6]
    for j in top:
        o=[]
        for k,(x,y) in sets.items():
            o.append('%s %.2f'%(k,roc_auc_score(np.r_[np.zeros(len(x)),np.ones(len(y))],np.r_[H[x][:,j],H[y][:,j]])))
        print('L%d n%-5d'%(L,j),' | '.join(o),'| act clean mean %.2f sd %.2f, big %.2f, small %.2f, F2 clean %.2f, F2 big %.2f'%(H[a][:,j].mean(),H[a][:,j].std(),H[b][:,j].mean(),H[sets['F1small'][1]][:,j].mean(),H[sets['F2'][0]][:,j].mean(),H[sets['F2'][1]][:,j].mean()))
H=np.load(D+'/mlp22.npy'); j=4847
for k,(x,y) in sets.items(): print(k,'n4847 clean',np.round(np.quantile(H[x][:,j],[.1,.5,.9]),2),'corrupt',np.round(np.quantile(H[y][:,j],[.1,.5,.9]),2),'paired win %.2f'%(H[y][:,j]>H[x][:,j]).mean())
