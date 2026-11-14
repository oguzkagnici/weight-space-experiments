# Hypothesis: late options get less consideration in the thinking: the sweep stops or thins out before reaching them, so a late gold is missed (lower accuracy, more cuts).
import json, numpy as np, collections, re
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
L='ABCDEFGHIJ'; Q=questions()
R=[];seen=set()
for f in ('passk_gen008champ_cap8192_det1.jsonl','trace_bank_gen008champ_cap8192_det1.jsonl'):
    for l in open(S+f):
        r=json.loads(l);k=(r['id'],r.get('sample',0))
        if k not in seen and len(Q[r['id']]['options'])==10: seen.add(k);R.append(r)
print('traces (10 options)',len(R))
def mentions(th):
    c=np.zeros(10)
    for m in re.finditer(r'(?:\(([A-J])\)|\b[Oo]ption ([A-J])\b|(?:^|\n)\s*\*{0,2}([A-J])[\):\.]|\b([A-J]):)',th):
        x=[g for g in m.groups() if g][0]; c[L.index(x)]+=1
    return c
M=np.array([mentions(r['think']) for r in R]); gold=np.array([L.index(r['gold']) for r in R]); cut=np.array([r['cut'] for r in R]); cor=np.array([r['correct'] for r in R])
print('mean mentions by letter  ',np.round(M.mean(0),1))
print('share of traces with >=1 ',np.round((M>0).mean(0),2))
print('share with >=2           ',np.round((M>1).mean(0),2))
ng=np.array([[M[i,k] for i in range(len(R)) if gold[i]!=k] for k in range(1)],dtype=object)
print('non-gold letters only, mean mentions:',np.round([np.mean([M[i,k] for i in range(len(R)) if gold[i]!=k]) for k in range(10)],1))
gm=M[np.arange(len(R)),gold]
for nm,m in (('right',cor==1),('wrong fin',(cor==0)&(cut==0)),('cut',cut==1)):
    print('%-9s n %3d gold mentions mean %.1f, zero %.2f | by gold A-E: mean %.1f zero %.2f (n %d); F-J: mean %.1f zero %.2f (n %d)'%(nm,m.sum(),gm[m].mean(),(gm[m]==0).mean(),gm[m&(gold<5)].mean(),(gm[m&(gold<5)]==0).mean(),(m&(gold<5)).sum(),gm[m&(gold>=5)].mean(),(gm[m&(gold>=5)]==0).mean(),(m&(gold>=5)).sum()))
# first mention position (fraction of thinking) by letter
def first(th):
    f=np.full(10,np.nan)
    for m in re.finditer(r'(?:\(([A-J])\)|\b[Oo]ption ([A-J])\b|(?:^|\n)\s*\*{0,2}([A-J])[\):\.]|\b([A-J]):)',th):
        x=L.index([g for g in m.groups() if g][0])
        if np.isnan(f[x]): f[x]=m.start()/max(1,len(th))
    return f
F=np.array([first(r['think']) for r in R]); print('median first-mention position by letter',np.round(np.nanmedian(F,0),2))
# first letter the thinking names as a candidate answer in any way = first mentioned letter overall
fm=np.nanargmin(np.where(np.isnan(F),9,F),1); print('first-mentioned letter counts',np.bincount(fm,minlength=10))
# tokens by gold position (finished)
tk=np.array([r['tokens'] for r in R]); print('mean tokens finished: gold A-E %.0f F-J %.0f'%(tk[(cut==0)&(gold<5)].mean(),tk[(cut==0)&(gold>=5)].mean()))
