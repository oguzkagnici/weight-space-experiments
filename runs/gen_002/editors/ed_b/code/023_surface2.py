# Hypothesis: (re-check of 022 with tie-aware ranks and letter-controlled baseline) the model avoids options that overlap lexically with the question, gold does not.
import numpy as np, json, re
from scipy.stats import rankdata
from wsp.editor import questions
qs=questions(); rows=[json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')]
def words(s): return set(re.findall(r'[a-z]{4,}',s.lower()))
res={'gold':[],'pred':[],'pred_wrong':[],'shuffled_pred':[]}; rng=np.random.default_rng(0); P=[]
for r in rows:
    q=qs[r['id']]; opts=q['options']; k=len(opts); qw=words(q['question'])
    O=np.array([len(words(o)&qw)/(len(words(o))+1) for o in opts]); L=np.array([len(o) for o in opts],float)
    if O.max()==O.min(): continue
    ro=(rankdata(O)-1)/(k-1); rl=(rankdata(L)-1)/(k-1)
    res['gold'].append((ro[r['answer']],rl[r['answer']])); res['pred'].append((ro[r['pred']],rl[r['pred']]))
    if not r['correct']: res['pred_wrong'].append((ro[r['pred']],rl[r['pred']]))
    P.append(np.array(r['probs'])[:k]@ro)
for nm,v in res.items():
    if v: v=np.array(v); print(nm,len(v),'overlap rank %.3f  length rank %.3f'%tuple(v.mean(0)))
print('prob-weighted overlap rank %.3f'%np.mean(P))
