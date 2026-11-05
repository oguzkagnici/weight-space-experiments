# Hypothesis: a "conclusion just stated" direction d in the L18 MLP input (x-hat space), from mean(concl) - mean(ctrl) breaks, projects much higher at conclusion breaks than at control breaks under cross-fitting; this is the gate we would wire into n5407's up projection.
import json, numpy as np
items=json.load(open('work/read1_items.json'))
xh=np.load('work/l18_xhat.npy')
qids=sorted(set(it['id'] for it in items)); rng=np.random.default_rng(0); fold={q:rng.integers(2) for q in qids}
f=np.array([fold[it['id']] for it in items]); isc=np.array([it['kind']=='concl' for it in items])
grp=np.array(['%s_%s_%d'%(it['kind'],it['label'] if it['kind']=='concl' else 'ctrl',it['cut']) for it in items])
def mkdir(tr):
    mu_all=xh[tr].mean(0)
    d=xh[tr&isc].mean(0)-xh[tr&~isc].mean(0)
    d=d/np.linalg.norm(d); return d
P=np.zeros(len(items))
for k in [0,1]:
    d=mkdir(f!=k); P[f==k]=xh[f==k]@d
for gname in sorted(set(grp)):
    m=grp==gname; print(gname,m.sum(),'proj mean %.2f sd %.2f'%(P[m].mean(),P[m].std()))
d=mkdir(np.ones(len(items),bool)); np.save('work/d_concl_l18.npy',d)
print('full-data d: conc %.2f ctrl %.2f; |xhat| mean %.1f'%((xh[isc]@d).mean(),(xh[~isc]@d).mean(),np.linalg.norm(xh,axis=1).mean()))
import torch
from wsp.editor import load
w3=load('model.layers.18.feed_forward.w3.weight')[5407].numpy(); w1=load('model.layers.18.feed_forward.w1.weight')[5407].numpy()
print('cos(d,w3) %.3f cos(d,w1) %.3f |w3| %.3f'%(d@w3/np.linalg.norm(w3),d@w1/np.linalg.norm(w1),np.linalg.norm(w3)))
