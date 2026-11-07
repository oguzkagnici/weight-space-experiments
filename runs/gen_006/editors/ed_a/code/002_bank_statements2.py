# Hypothesis: a broader "answer ... X" pattern finds the ~18 cut traces with stated answers; tally hedged vs firm and correctness.
import json, re
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
pat=re.compile(r'answer(?: is|:| would be| should be| likely| might be| maybe| =)?(?: likely| probably| maybe| perhaps)?(?: is)?:?\s*(?:option\s*)?\(?\**([A-J])(?:\)|\b)')
HEDGE=re.compile(r'\b(likely|maybe|perhaps|probably|might|could|guess|seems|I think|possibly|or \(?[A-J]\b)', re.I)
rows=[]
for r in B:
    th=r['think']; n=len(th)
    ms=[m for m in pat.finditer(th)]
    # first statement
    if not ms: continue
    for m in ms:
        ls=th.rfind('\n',0,m.start())+1; le=th.find('\n',m.end()); le=n if le<0 else le
        line=th[ls:le]
        rows.append(dict(id=r['id'],cut=r['cut'],correct=r['correct'],pos=m.start()/n,let=m.group(1),gold=r['gold'],hedge=bool(HEDGE.search(line)),line=line[:140]))
import collections
for cut in (0,1):
    ids=set(x['id'] for x in rows if x['cut']==cut)
    print('cut',cut,'traces with statement',len(ids))
# per cut trace, first statement
print('--- cut traces, first statement per trace')
seen=set(); stats=collections.Counter()
for x in rows:
    if x['cut']!=1 or x['id'] in seen: continue
    seen.add(x['id'])
    stats[(x['hedge'],x['let']==x['gold'])]+=1
    print(f"{x['id'][8:]:>6} {x['pos']:.2f} {x['let']}{'*' if x['let']==x['gold'] else ' '} {'H' if x['hedge'] else 'F'} | {x['line'][:120]}")
print(stats)
print('--- finished traces, first statement: hedged?')
seen=set(); st2=collections.Counter()
for x in rows:
    if x['cut']!=0 or x['id'] in seen: continue
    seen.add(x['id']); st2[(x['hedge'],x['let']==x['gold'])]+=1
print(st2)
# all statements (not just first) in cut traces
st3=collections.Counter((x['hedge'],x['let']==x['gold']) for x in rows if x['cut']==1)
print('all stmts cut',st3)
st4=collections.Counter((x['hedge'],x['let']==x['gold']) for x in rows if x['cut']==0)
print('all stmts finished',st4)
