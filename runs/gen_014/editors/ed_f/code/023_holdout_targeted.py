# Hypothesis: on the holdout traces the edit first changes the text at or after the first option-judging paragraph (its target), and raises stem re-statement in the judging region.
import json, sys, importlib, numpy as np, re
sys.path.insert(0,'code'); b=importlib.import_module('005_build_ins'); f3=importlib.import_module('003_stem_restate')
from wsp.editor import *
bank=sys.argv[1]; files=sys.argv[2:]; B={}; Ed={}
for l in open(bank):
    r=json.loads(l); B[r['id']]=r
for f in files:
    for l in open(f):
        r=json.loads(l); Ed[r['id']]=r
keys=[k for k in Ed if k in B]; rel=[]; before=after=same=0
for k in keys:
    a,bb=B[k]['raw'],Ed[k]['raw']
    if a==bb: same+=1; continue
    n=next((i for i,(x,y) in enumerate(zip(a,bb)) if x!=y),min(len(a),len(bb)))
    T,off,think,early,late=b.cuts(B[k])
    if early is None: continue
    e=early+7 if B[k]['raw'].startswith('<think>') else early
    if n<e: before+=1
    else: after+=1
    rel.append(n/max(1,e))
print('pairs %d | identical %d | first difference BEFORE the first judging paragraph %d, at/after it %d | median (first difference / start of judging) %.2f'%(len(keys),same,before,after,np.median(rel)))
F0=[];F1=[]
for k in keys:
    if B[k]['cut'] or Ed[k]['cut']: continue
    x,y=f3.features(B[k]),f3.features(Ed[k])
    if x and y: F0.append(x); F1.append(y)
for key in ('so','g3','quote','mark','n'):
    d=np.array([y[key]-x[key] for x,y in zip(F0,F1)]); print('  judging-region %-5s parent %.2f edited %.2f diff %+.2f (SE %.2f, n %d both-finished)'%(key,np.mean([x[key] for x in F0]),np.mean([y[key] for y in F1]),d.mean(),d.std(ddof=1)/np.sqrt(len(d)),len(d)))
