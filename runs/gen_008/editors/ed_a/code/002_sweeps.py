# Hypothesis: cut traces start more option sweeps (2nd/3rd sweep) than finished ones; count sweep starts and their position.
import json, re, collections, numpy as np
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
SW=re.compile(r"^(?:Now,? |So |Thus,? |But |Better |We )?(?:let'?s|let’s|let us|we(?:'ll| will| need to| should| can)?)\s+(?:re-?)?(?:examine|analy[sz]e|evaluate|go through|check|consider|review|look at|assess|test|parse|re-?evaluate|re-?examine|reconsider|scan)\s+(?:each|all|the|every|them|options|again)",re.I)
FIRST=re.compile(r"^(?:[-*]\s*)?(?:\*\*)?(?:Option\s+A\b|\(A\)|A[.:)]\s)")
rows=[]
for b in B:
    t=b['think']; paras=[]; pos=0
    for m in re.finditer(r'(?:^|\n\n)([^\n].*?)(?=\n\n|$)', t, re.S):
        paras.append((m.start(1), m.group(1)))
    L=len(t); starts=[]
    for (s,p) in paras:
        ps=p.strip()
        if SW.match(ps) or FIRST.match(ps):
            if not starts or s-starts[-1]>300: starts.append(s)
    rows.append(dict(id=b['id'],cat=b['category'],cut=b['cut'],cor=b['correct'],n=len(starts),fr=[round(s/L,2) for s in starts]))
for c in (0,1):
    ns=[r['n'] for r in rows if r['cut']==c]
    print('cut' if c else 'fin', len(ns), 'mean sweeps', np.mean(ns), collections.Counter(ns))
print('cut traces with >=2 sweeps:', sum(1 for r in rows if r['cut'] and r['n']>=2), ' finished with >=2:', sum(1 for r in rows if not r['cut'] and r['n']>=2), 'finished>=2 correct', sum(1 for r in rows if not r['cut'] and r['n']>=2 and r['cor']))
for r in rows:
    if r['n']>=2: print(r['id'],r['cat'][:6],'cut' if r['cut'] else ('ok ' if r['cor'] else 'bad'),r['n'],r['fr'])
json.dump(rows,open('work/sweeps.json','w'))
