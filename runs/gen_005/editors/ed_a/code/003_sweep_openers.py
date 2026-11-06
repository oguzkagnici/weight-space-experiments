# Hypothesis: repeat sweeps are introduced by a stereotyped opener line ("Let's examine each option ...") that is a nameable decision point.
import json, re, numpy as np, collections
from wsp.editor import *
Q = questions()
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
pat=re.compile(r"(each|every|all|remaining|other) (of the )?(option|answer|choice|candidate)|go through (the )?(option|choice)|list (them|the options)|evaluate (the )?options|options? one by one|examine (the )?options|consider (the )?options", re.I)
cnt=collections.Counter(); per=[]
for id_,r in rows.items():
    t=r['think']; lines=[l for l in t.split('\n')]
    hits=[]; pos=0
    for l in lines:
        if pat.search(l) and len(l)<200: hits.append((pos/len(t), l.strip()[:90]))
        pos+=len(l)+1
    per.append((r['cut'],r['correct'],len(hits),r['tokens'],r['category']))
    for k,(p,l) in enumerate(hits):
        cnt[(r['cut'], min(k,3))]+=1
print(sorted(cnt.items()))
for cut in (0,1):
    s=[p for p in per if p[0]==cut]
    print('cut',cut,'opener lines per trace %.2f'%np.mean([p[2] for p in s]),'per 1k tok %.2f'%(1000*np.mean([p[2]/p[3] for p in s])))
# show examples from cut traces
import random; random.seed(0)
ex=[]
for id_,r in rows.items():
    t=r['think']
    for m in pat.finditer(t):
        a=t.rfind('\n',0,m.start())+1; b=t.find('\n',m.end()); 
        ex.append((r['cut'],t[a:b if b>0 else None][:110]))
for c,l in random.sample(ex,40): print(c,l.encode('ascii','replace').decode())
