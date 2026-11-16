# Hypothesis: most finished-wrong verbal questions have an identifiable decisive stem phrase (cue) separating gold from the wrong pick; list them for hand annotation.
import json, sys, importlib, collections
sys.path.insert(0,'code'); m=importlib.import_module('001_overview')
from wsp.editor import *
Q=questions(); rows=m.load_rows(); QUANT={'business','chemistry','physics','math','engineering'}
asc=lambda t:str(t).encode('ascii','replace').decode()
byq=collections.defaultdict(list)
for r in rows: byq[r['id']].append(r)
n=0
for q,rs in sorted(byq.items()):
    wr=[r for r in rs if not r['cut'] and not r['correct'] and r['pred']]
    if not wr: continue
    n+=1; qq=Q[q]; L='ABCDEFGHIJ'
    pat=''.join('-' if r['cut'] else (r['pred'] or '?') for r in sorted(rs,key=lambda r:r['sample']))
    print('\n##',q,qq['category'],'gold',qq['answer'],'preds',pat,'toks',[r['tokens'] for r in wr])
    print('STEM:',asc(qq['question'])[:1400])
    print('GOLD',qq['answer'],':',asc(qq['options'][qq['answer_index']])[:200])
    for p in sorted(set(r['pred'] for r in wr)):
        if p in L and L.index(p)<len(qq['options']): print('WRONG',p,':',asc(qq['options'][L.index(p)])[:200])
print(n)
