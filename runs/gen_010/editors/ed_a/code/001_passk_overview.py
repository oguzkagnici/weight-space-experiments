# Hypothesis: mixed questions in the pass@4 bank show which reasoning step separates right from wrong samples; overview first.
import json, collections
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in rows: by[r['id']].append(r)
print(len(rows), len(by))
for qid,rs in sorted(by.items(), key=lambda kv: sum(r['correct'] for r in kv[1])):
    rs=sorted(rs,key=lambda r:r['sample'])
    nc=sum(r['correct'] for r in rs)
    s=' '.join(f"{r['pred'] or '-'}{'*' if r['correct'] else ''}{'c' if r['cut'] else ''}:{r['tokens']}" for r in rs)
    print(f"{qid:16s} {rs[0]['category'][:10]:10s} gold={rs[0]['gold']} n={nc} | {s}")
