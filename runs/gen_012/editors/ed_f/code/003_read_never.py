# Hypothesis: some never-right questions are recurring reasoning slips (negation, calc->option mapping) rather than missing knowledge; print compact views.
import json, collections, sys
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
pk=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
Q=questions()
by=collections.defaultdict(list)
for r in pk: by[r['id']].append(r)
ids=sys.argv[1:]
def a(s): return s.encode('ascii','replace').decode()
for q in ids:
    rs=by[q]; qq=Q[q]; g=qq['answer']
    print('=====',q,qq['category'],'gold',g,'preds',''.join(str(r['pred']) if not r['cut'] else '-' for r in rs))
    print('Q:',a(qq['question'][-350:]))
    L='ABCDEFGHIJ'
    print(' GOLD (%s) %s'%(g,a(qq['options'][L.index(g)][:140])))
    fin=[r for r in rs if not r['cut'] and not r['correct']]
    if fin:
        r=fin[0]; p=r['pred']
        if p in L and L.index(p)<len(qq['options']): print(' PICK (%s) %s'%(p,a(qq['options'][L.index(p)][:140])))
        print(' END:',a(r['think'][-420:]).replace('\n',' | '))
