# Hypothesis: the maj@4 gain over pass@1 is partly cut-avoidance; measure content-only headroom on finished samples.
import json, collections, numpy as np
S='/home/user/weight-space-experiments/runs/shared/'
pk=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in pk: by[r['id']].append(r)
acc=[];maj=[];n=0;rows=[]
for q,rs in by.items():
    fin=[r for r in rs if not r['cut']]
    if not fin: continue
    a=np.mean([r['correct'] for r in fin])
    c=collections.Counter(r['pred'] for r in fin); top=max(c.values()); tl=[k for k,v in c.items() if v==top]
    m=np.mean([k==rs[0]['gold'] for k in tl])
    acc.append(a);maj.append(m)
    rows.append((q,rs[0]['category'],rs[0]['gold'],''.join(str(r['pred']) if not r['cut'] else '-' for r in rs),[r['tokens'] for r in rs]))
print('questions with a finished sample',len(acc),'mean finished acc %.3f  maj-of-finished %.3f'%(np.mean(acc),np.mean(maj)))
mixed=[r for r in rows if 0<sum(ch==r[2] for ch in r[3])<sum(ch!='-' for ch in r[3])]
print('mixed among finished',len(mixed))
for r in mixed: print(r)
never=[r for r in rows if sum(ch==r[2] for ch in r[3])==0]
print('never right',len(never)); 
for r in never: print(r[:4])
# does the wrong answer agree across samples (systematic) on never-right questions?
