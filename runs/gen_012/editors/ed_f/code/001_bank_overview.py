# Hypothesis: none yet; overview of the 8,192 banks (counts by category, finished/wrong) to pick wrong finished traces to read.
import json, collections
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
pk=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
tb=[json.loads(l) for l in open(S+'trace_bank_gen008champ_cap8192_det1.jsonl')]
print(len(pk),len(tb),pk[0].keys())
Q=questions()
for name,rows in (('passk',pk),('bank',tb)):
    c=collections.Counter()
    for r in rows:
        k='cut' if r['cut'] else ('right' if r['correct'] else 'wrong')
        c[(r['category'],k)]+=1
    cats=sorted({k[0] for k in c})
    print(name)
    for ca in cats: print('  %-18s right %3d wrong %3d cut %3d'%(ca,c[(ca,'right')],c[(ca,'wrong')],c[(ca,'cut')]))
ids_pk={r['id'] for r in pk}; ids_tb={r['id'] for r in tb}
print('bank ids not in passk',len(ids_tb-ids_pk))
import numpy as np
print('tokens mean',np.mean([r['tokens'] for r in tb]))
