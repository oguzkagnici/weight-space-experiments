# Hypothesis (step 2): CUE insertion raises log P(gold) - log P(wrong pick) at the forced conclusion more than the NON-cue control; CUE+INF (oracle implication) shows whether the missing piece is the inference.
# usage: 008_ins_analysis.py <pass:dir> ...   e.g. p1:work/fwd_01 p2:work/fwd_02
import json, sys, collections, numpy as np
from wsp.editor import *
L='ABCDEFGHIJ'
R={}
for a in sys.argv[1:]:
    name,D=a.split(':'); meta=json.load(open('work/%s_meta.json'%name)); ids=np.load(D+'/topk_ids.npy'); lp=np.load(D+'/topk_logprobs.npy')
    assert len(meta)==len(ids)
    for n,mt in enumerate(meta):
        if mt['kind']!='ins': continue
        d={int(i):float(l) for i,l in zip(ids[n],lp[n])}; fl=lp[n].min()-0.5
        v=np.array([d.get(int(t),fl) for t in LETTER_IDS[:10]]); mt=dict(mt); mt['lp']=v; mt['mass']=float(np.exp(v).sum())
        R[(mt['id'],mt['sample'],mt['cut'],mt['arm'])]=mt
def val(mt,what):
    v=mt['lp']; g=L.index(mt['gold']); w=L.index(mt['wrong'])
    if what=='margin': return v[g]-v[w]
    if what=='pgold': return float(np.exp(v[g])/np.exp(v).sum())
    if what=='pwrong': return float(np.exp(v[w])/np.exp(v).sum())
    if what=='top': return float(np.argmax(v)==g)
def cmp(cut,status,a,b):
    keys=[k for k in R if k[2]==cut and k[3]==a and R[k]['status'] in status and (k[0],k[1],cut,b) in R]
    if not keys: return
    out=[]
    for what in ('margin','pgold','pwrong','top'):
        da=np.array([val(R[k],what) for k in keys]); db=np.array([val(R[(k[0],k[1],cut,b)],what) for k in keys]); q=np.array([k[0] for k in keys])
        dq=np.array([(da-db)[q==x].mean() for x in sorted(set(q))]); se=dq.std(ddof=1)/np.sqrt(len(dq)) if len(dq)>1 else float('nan')
        out.append('%s %s %.3f vs %s %.3f: diff %+.3f (SE %.3f, z %+.1f; questions up %d/%d)'%(what,a,da.mean(),b,db.mean(),dq.mean(),se,dq.mean()/se if se>0 else 0,(dq>0).sum(),len(dq)))
    print('[%s cut, %s samples, n %d, %d questions] %s - %s'%(cut,'/'.join(status),len(keys),len(set(k[0] for k in keys)),a,b))
    for o in out: print('    '+o)
print('texts',len(R),'mean letter mass %.3f'%np.mean([r['mass'] for r in R.values()]))
for cut in ('late','early'):
    for st in (('wrong',),('cut',),('right',),('wrong','cut','right')):
        for a,b in (('CUE','NON'),('INF','NON'),('INF','CUE'),('NON','BASE'),('CUE','BASE')): cmp(cut,st,a,b)
# per-question table (wrong samples, late cut)
print('\nper question, wrong samples: margin log P(gold)-log P(wrong pick) [late: BASE NON CUE INF | early: NON CUE INF]')
for q in sorted(set(k[0] for k in R)):
    row=[]
    for cut,arms in (('late',('BASE','NON','CUE','INF')),('early',('NON','CUE','INF'))):
        for a in arms:
            v=[val(R[k],'margin') for k in R if k[0]==q and k[2]==cut and k[3]==a and R[k]['status']=='wrong']
            row.append('%6.2f'%np.mean(v) if v else '   -- ')
        row.append('|')
    print(q,' '.join(row))
