# Hypothesis: none (bookkeeping). Split the 96 trace-bank questions into DEV 64 / HOLDOUT 32, stratified by cut, seed 5.
import json, random
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')]
random.seed(5)
cut=[r['id'] for r in rows if r['cut']]; fin=[r['id'] for r in rows if not r['cut']]
random.shuffle(cut); random.shuffle(fin)
hold=cut[:17]+fin[:15]; dev=[i for i in [r['id'] for r in rows] if i not in hold]
json.dump({'dev':dev,'holdout':hold},open('work/split.json','w'))
R={r['id']:r for r in rows}
for n,s in (('dev',dev),('hold',hold)):
    print(n,len(s),'cut',sum(R[i]['cut'] for i in s),'acc %.3f'%(sum(R[i]['correct'] for i in s)/len(s)))
