# Hypothesis: selection quality depends on option index / text length (e.g. sliding-window 256 limits late options' access to the question).
import numpy as np, json
from wsp.editor import questions
rows=[json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')]; qs=questions()
k=np.array([len(qs[r['id']]['options']) for r in rows]); g=np.array([r['answer'] for r in rows]); c=np.array([r['correct'] for r in rows]); n=np.array([r['n_tokens'] for r in rows])
lp=np.array([r['logp_correct'] for r in rows])
for lo,hi in [(0,150),(150,220),(220,300),(300,2000)]:
    m=(n>=lo)&(n<hi); print(f'len {lo}-{hi}: n {m.sum()} acc {c[m].mean():.3f} logp {lp[m].mean():.2f}')
for L in range(10):
    m=(g==L)&(k==10); print('gold',  'ABCDEFGHIJ'[L], m.sum(), f'acc {c[m].mean():.2f}')
H=np.load('work/p3_heads_dl.npy')[12]
acc12=np.array([np.argmax(H[i,:k[i]])==g[i] for i in range(len(rows))])
print('H12-alone acc by gold half (10-opt): A-E %.3f F-J %.3f'%(acc12[(k==10)&(g<5)].mean(),acc12[(k==10)&(g>=5)].mean()))
print('final acc by gold half: A-E %.3f F-J %.3f'%(c[(k==10)&(g<5)].mean(),c[(k==10)&(g>=5)].mean()))
