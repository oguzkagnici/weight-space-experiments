# Hypothesis: finished-answer accuracy falls with the gold option's position (question level), in every bank and within categories; also check cut rate and the unedited model.
import json, numpy as np, collections
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
L='ABCDEFGHIJ'; Q=questions()
def load(fs):
    R=[]; seen=set()
    for f in fs:
        for l in open(S+f):
            r=json.loads(l); k=(r['id'],r.get('sample',0))
            if k not in seen: seen.add(k); R.append(r)
    return R
def rep(name,R):
    fin=collections.defaultdict(list); cut=collections.defaultdict(list)
    for r in R:
        cut[r['id']].append(r['cut'])
        if not r['cut']: fin[r['id']].append(r['correct'])
    q=list(fin); gp=np.array([L.index(Q[i]['answer']) for i in q]); no=np.array([len(Q[i]['options']) for i in q]); ac=np.array([np.mean(fin[i]) for i in q])
    cat=np.array([Q[i]['category'] for i in q])
    rel=gp/(no-1)
    # category-adjusted: subtract category mean acc
    adj=ac-np.array([ac[cat==c].mean() for c in cat])
    r1=np.corrcoef(gp,ac)[0,1]; r2=np.corrcoef(gp,adj)[0,1]
    rng=np.random.default_rng(0); perm=[np.corrcoef(rng.permutation(gp),ac)[0,1] for _ in range(20000)]
    print('%-22s nq %d acc by gold idx 0-1 %.2f(n%d) 2-4 %.2f(n%d) 5-7 %.2f(n%d) 8-9 %.2f(n%d) | r %.3f (perm p1 %.4f) cat-adj r %.3f'%(name,len(q),
      ac[gp<2].mean(),(gp<2).sum(),ac[(gp>=2)&(gp<5)].mean(),((gp>=2)&(gp<5)).sum(),ac[(gp>=5)&(gp<8)].mean(),((gp>=5)&(gp<8)).sum(),ac[gp>=8].mean(),(gp>=8).sum(),r1,np.mean(np.array(perm)<=r1),r2))
    qa=list(cut); g2=np.array([L.index(Q[i]['answer']) for i in qa]); cr=np.array([np.mean(cut[i]) for i in qa])
    print('    cut rate gold A-E %.2f F-J %.2f'%(cr[g2<5].mean(),cr[g2>=5].mean()))
rep('champ 8192 (passk+bank)',load(['passk_gen008champ_cap8192_det1.jsonl','trace_bank_gen008champ_cap8192_det1.jsonl']))
rep('champ 3072',load(['trace_bank_gen008champ_cap3072_det1.jsonl']))
rep('base 3072 det1',load(['trace_bank_base_cap3072_det1.jsonl']))
rep('base 3072 (nondet)',load(['trace_bank_base_cap3072.jsonl']))
rep('gen5 3072',load(['trace_bank_gen005champ_cap3072_det1.jsonl']))
# gold index distribution in all practice questions (10-option)
g=np.bincount([q['answer_index'] for q in Q.values() if len(q['options'])==10],minlength=10); print('practice gold idx counts (10 options)',g)
