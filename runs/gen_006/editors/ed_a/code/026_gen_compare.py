# Hypothesis: with edit v2a, edited traces are byte-identical to the parent up to </think> in most questions, answers become
# '"The answer is (X)".' + end, and traces the parent cut inside its answer now finish with the think's answer.
import json, sys, numpy as np
k=int(sys.argv[1]); D=f'work/fwd_{k:02d}/'
O={json.loads(l)['id']:json.loads(l) for l in open(D+f'gen_{k}_original.jsonl')}
E={json.loads(l)['id']:json.loads(l) for l in open(D+f'gen_{k}_edited.jsonl')}
print(open(D+'summary.json').read()[:1500] if __import__('os').path.exists(D+'summary.json') else '')
same_think=0; stats=[]
for q in O:
    o,e=O[q],E[q]; a,b=o['raw'],e['raw']
    i=next((t for t in range(min(len(a),len(b))) if a[t]!=b[t]),min(len(a),len(b)))
    ca=a.find('</think>'); 
    where='identical' if a==b else ('after </think>' if ca>=0 and i>=ca else f'in think at {i} ({i/max(1,len(a)):.2f})')
    if a==b or (ca>=0 and i>=ca): same_think+=1
    flag={(0,1):'GAIN',(1,0):'LOSS'}.get((o['correct'],e['correct']),'')
    print(f"{q[8:]:>6} {o['category'][:6]:6} orig {o['pred']}/{o['gold']} c{o['correct']} cut{o['cut']} {o['tokens']:4d} | edit {e['pred']} c{e['correct']} cut{e['cut']} {e['tokens']:4d} | {where} {flag} | content {e['content'][:40]!r}")
print('think identical (or diverge only after </think>):',same_think,'of',len(O))
