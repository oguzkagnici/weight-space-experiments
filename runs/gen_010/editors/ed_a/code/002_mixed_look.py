# Hypothesis: in mixed questions, wrong samples fail at a recognisable step (misread question, wrong criterion); print question and conclusion snippets.
import json, collections, sys
from wsp.editor import questions
Q=questions()
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in rows: by[r['id']].append(r)
for qid in sys.argv[1:]:
    q=Q[qid]; print('='*80); print(qid, q['category'], 'gold', q['answer'])
    print(q['question'][:700].encode('ascii','replace').decode())
    for i,o in enumerate(q['options']): print(' ', 'ABCDEFGHIJ'[i], o[:150].encode('ascii','replace').decode())
    for r in sorted(by[qid],key=lambda r:r['sample']):
        t=r['think']
        print(f"--- s{r['sample']} pred={r['pred']} correct={r['correct']} tok={r['tokens']} len={len(t)}")
        print('   START:', t[7:400].replace('\n',' | ').encode('ascii','replace').decode())
        print('   END:', t[-500:].replace('\n',' | ').encode('ascii','replace').decode())
