# Final measurement: paired comparison of edited generate rows with the parent bank rows (holdout48 at 8,192): overall, both-finish, first difference; hypothesis: the edit changes conclusions on questions both models finish.
import json, sys, numpy as np, collections
from wsp.editor import *
bank=sys.argv[1]; files=sys.argv[2:]
B={}
for l in open(bank):
    r=json.loads(l); B[(r['id'],r.get('sample',0))]=r
Ed={}
for f in files:
    for l in open(f):
        r=json.loads(l); Ed[(r['id'],r.get('sample',0))]=r
keys=[k for k in Ed if k in B]; print('pairs',len(keys),'edited rows',len(Ed))
L='ABCDEFGHIJ'
p=np.array([B[k]['correct'] for k in keys],float); e=np.array([Ed[k]['correct'] for k in keys],float)
pc=np.array([B[k]['cut'] for k in keys]); ec=np.array([Ed[k]['cut'] for k in keys]); g=np.array([L.index(B[k]['gold']) for k in keys])
qid=np.array([k[0] for k in keys])
def rep(tag,m):
    d=(e-p)[m]
    # cluster by question when several samples per question
    qs=sorted(set(qid[m])); dq=np.array([d[qid[m]==q].mean() for q in qs]); se=dq.std(ddof=1)/np.sqrt(len(dq)) if len(dq)>1 else float('nan')
    print('%-22s n %3d (q %2d) | %.1f -> %.1f | diff %+.2f (SE %.2f) | +%d/-%d'%(tag,m.sum(),len(qs),100*p[m].mean(),100*e[m].mean(),100*dq.mean() if len(qs)!=m.sum() else 100*d.mean(),100*se,(d>0).sum(),(d<0).sum()))
al=np.ones(len(keys),bool)
rep('overall',al); rep('both finish',(pc==0)&(ec==0)); rep('gold A-E',g<5); rep('gold F-J',g>=5); rep('both finish, gold A-E',(pc==0)&(ec==0)&(g<5)); rep('both finish, gold F-J',(pc==0)&(ec==0)&(g>=5))
print('cut parent %d edited %d | cut->fin %d (right %d) fin->cut %d (were right %d)'%(pc.sum(),ec.sum(),((pc==1)&(ec==0)).sum(),e[(pc==1)&(ec==0)].sum(),((pc==0)&(ec==1)).sum(),p[(pc==0)&(ec==1)].sum()))
print('mean tokens parent %.0f edited %.0f'%(np.mean([B[k]['tokens'] for k in keys]),np.mean([Ed[k]['tokens'] for k in keys])))
fd=[]
for k in keys:
    a,b=B[k]['raw'],Ed[k]['raw']; n=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b))); fd.append(1.0 if a==b else n/max(1,len(a)))
print('identical traces %d, median first difference at %.1f%% of the parent trace'%(sum(f==1.0 for f in fd),100*np.median(fd)))
bf=(pc==0)&(ec==0)
print('pred letters (finished): parent',np.bincount([L.index(B[k]['pred']) for k in keys if not B[k]['cut'] and B[k]['pred'] in L],minlength=10),'edited',np.bincount([L.index(Ed[k]['pred']) for k in keys if not Ed[k]['cut'] and Ed[k]['pred'] in L],minlength=10),'gold',np.bincount(g,minlength=10))
print('changed answers among both-finish: %d of %d'%(sum(B[k]['pred']!=Ed[k]['pred'] for k,m in zip(keys,bf) if m),bf.sum()))
