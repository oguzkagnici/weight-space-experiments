# DEV generate for H2: 6 passk questions x 4 samples at 8,192 (parent cached): 4 late-gold questions with wrong/mixed finished answers, 2 early-gold always-right controls.
import json, collections, numpy as np
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
pk=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in pk: by[r['id']].append(r)
L='ABCDEFGHIJ'
ctrl=[(q,rs[0]['gold'],int(np.mean([r['tokens'] for r in rs]))) for q,rs in by.items() if all(r['correct'] for r in rs) and rs[0]['gold'] in 'ABC' and rs[0]['category'] in ('law','health','psychology','economics','other','history','philosophy')]
print('early-gold always-right verbal controls',ctrl)
late=['mmlupro_6110','mmlupro_6800','mmlupro_2055','mmlupro_5510']
sel=late+[c[0] for c in sorted(ctrl,key=lambda c:c[2])[len(ctrl)//2-1:len(ctrl)//2+1]]
print(sel,[(by[q][0]['gold'],''.join(str(r['pred']) if not r['cut'] else '-' for r in by[q])) for q in sel])
json.dump({"type":"generate","model":"edited","questions":sel,"n":4,"cap":8192},open('work/req_dev.json','w'))
