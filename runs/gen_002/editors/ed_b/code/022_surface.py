# Hypothesis: the option belief leans on surface features (option length, lexical overlap with the question) more than the gold does -> a heuristic component to suppress.
import numpy as np, json, re
from wsp.editor import questions
qs=questions(); rows=[json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')]
def words(s): return set(re.findall(r'[a-z]{4,}',s.lower()))
stats={'len':[], 'ovl':[]}
out={'gold':{'len':[],'ovl':[]},'pred':{'len':[],'ovl':[]},'pred_wrong':{'len':[],'ovl':[]}}
for r in rows:
    q=qs[r['id']]; opts=q['options']; k=len(opts)
    L=np.array([len(o) for o in opts],float); qw=words(q['question'])
    O=np.array([len(words(o)&qw)/(len(words(o))+1) for o in opts])
    rl=(L.argsort().argsort())/(k-1); ro=(O.argsort().argsort())/(k-1)
    for nm,i in [('gold',r['answer']),('pred',r['pred'])]+([('pred_wrong',r['pred'])] if not r['correct'] else []):
        out[nm]['len'].append(rl[i]); out[nm]['ovl'].append(ro[i])
for nm in out: print(nm, 'mean length rank %.3f overlap rank %.3f (0.5 = neutral)'%(np.mean(out[nm]['len']),np.mean(out[nm]['ovl'])), len(out[nm]['len']))
# longest option rate
lg=np.mean([np.argmax([len(o) for o in qs[r['id']]['options']])==r['answer'] for r in rows]); lp=np.mean([np.argmax([len(o) for o in qs[r['id']]['options']])==r['pred'] for r in rows])
print('gold is longest %.3f, pred is longest %.3f'%(lg,lp))
