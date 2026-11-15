# Hypothesis: mixed pass@4 questions (right and wrong finished samples) expose step-level errors; list them.
import json, collections
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
pk=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in pk: by[r['id']].append(r)
Q=questions()
for q,rs in by.items():
    rs.sort(key=lambda r:r['sample'])
    nc=sum(r['correct'] for r in rs)
    s=''.join((str(r['pred']) if not r['cut'] else '-') for r in rs)
    tag='MIXED' if 0<nc<4 else ('never' if nc==0 else 'always')
    print(q,rs[0]['category'][:6],rs[0]['gold'],s,tag,[r['tokens'] for r in rs],len(Q[q]['options']))
