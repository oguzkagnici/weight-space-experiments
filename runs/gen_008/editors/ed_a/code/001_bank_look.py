# Hypothesis: cut law/health traces repeatedly restart full option sweeps; look at the paragraph-opening patterns.
import json, re, collections
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
print(len(B), sum(b['cut'] for b in B), sum(b['correct'] for b in B))
op=collections.Counter(); opcut=collections.Counter()
for b in B:
    paras=[p for p in b['think'].split('\n\n') if p.strip()]
    for p in paras:
        w=' '.join(p.strip().split()[:2])[:25]
        op[w]+=1
        if b['cut']: opcut[w]+=1
print([ (w,c,opcut[w]) for w,c in op.most_common(60)])
# print the paragraph openings of one cut law trace
for b in B:
    if b['category']=='law' and b['cut']:
        paras=[p for p in b['think'].split('\n\n') if p.strip()]
        print(b['id'], len(paras))
        for p in paras[:80]: print('   |', p[:90].replace('\n',' / ').encode('ascii','replace').decode())
        break
