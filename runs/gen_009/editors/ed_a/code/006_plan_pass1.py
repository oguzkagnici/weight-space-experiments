# Hypothesis: probe texts (trace prefix to an eval sweep end + forced "Hence answer is (") plus full non-sweep traces fit in the 200k-token read budget.
import json
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
S=json.load(open('work/sweeps.json'))
FORCE="Hence answer is ("
texts=[];meta=[]
ntok=lambda s: len(tok.encode(s,add_special_tokens=False))
for r in B:
    tt=trace_text(r); P=render_think(r['id'])
    assert tt.startswith(P) and r['think'].startswith('<think>')
    th=r['think'][len('<think>'):]
    assert tt[len(P):len(P)+200]==th[:200], r['id']
    ev=[x for x in S[r['id']] if x['eval']]
    if ev:
        sel=[ev[-1]] + ([ev[0]] if len(ev)>=2 else [])
        for x in sel:
            pos=len(P)+x['end']-len('<think>')+2
            assert tt[pos-2:pos]=='\n\n'
            texts.append(tt[:pos]+FORCE); meta.append(dict(id=r['id'],kind='probe',cut=r['cut'],correct=r['correct'],gold=r['gold'],which='last' if x is ev[-1] else 'first',nsw=len(ev)))
    else:
        texts.append(tt); meta.append(dict(id=r['id'],kind='full',cut=r['cut'],correct=r['correct'],gold=r['gold']))
n=[ntok(t) for t in texts]
import collections
c=collections.defaultdict(int)
for m,k in zip(meta,n): c[m['kind']]+=k
print(len(texts),sum(n),dict(c),max(n))
json.dump(dict(texts=texts,meta=meta,ntok=n),open('work/pass1_plan.json','w'))
