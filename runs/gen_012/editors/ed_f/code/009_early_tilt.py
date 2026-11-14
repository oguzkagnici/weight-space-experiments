# Hypothesis: wrong final picks are earlier options than gold more often than later (question-level sign test), beyond what the positions allow by chance.
import json, numpy as np, collections
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
L='ABCDEFGHIJ'; Q=questions()
R=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]+[json.loads(l) for l in open(S+'trace_bank_gen008champ_cap8192_det1.jsonl')]
seen=set(); by=collections.defaultdict(list)
for r in R:
    k=(r['id'],r.get('sample',0))
    if k in seen: continue
    seen.add(k)
    if not r['cut'] and r['pred'] in L and not r['correct']: by[r['id']].append(r)
e=l=0; exp=0; rows=[]
for q,rs in by.items():
    g=L.index(rs[0]['gold']); n=len(Q[q]['options'])
    fr=np.mean([L.index(r['pred'])<g for r in rs]); e+=fr; l+=1-fr
    exp+=g/(n-1)   # chance share of earlier options among wrong ones
    rows.append((q,rs[0]['category'],g,n,fr))
print('questions with a wrong finished answer',len(by),'earlier %.1f later %.1f; chance-expected earlier %.1f'%(e,l,exp))
# simulation SE under uniform wrong pick
rng=np.random.default_rng(0); sim=[sum(rng.random()<(g/(n-1)) for _,_,g,n,_ in rows) for _ in range(20000)]
print('p(sim >= obs) = %.4f'%np.mean(np.array(sim)>=e))
# by category
c=collections.defaultdict(lambda:[0,0,0])
for q,cat,g,n,fr in rows: c[cat][0]+=fr; c[cat][1]+=1-fr; c[cat][2]+=g/(n-1)
print({k:[round(x,1) for x in v] for k,v in c.items()})
# gold position effect on accuracy at question level (finished samples)
allr=collections.defaultdict(list)
seen=set()
for r in R:
    k=(r['id'],r.get('sample',0))
    if k in seen or r['cut']: continue
    seen.add(k); allr[r['id']].append(r['correct'])
gp=np.array([L.index(Q[q]['answer']) for q in allr]); ac=np.array([np.mean(v) for v in allr.values()])
print('n q',len(gp),'acc gold in A-E %.3f (n %d) vs F-J %.3f (n %d)'%(ac[gp<5].mean(),(gp<5).sum(),ac[gp>=5].mean(),(gp>=5).sum()), 'corr %.3f'%np.corrcoef(gp,ac)[0,1])
