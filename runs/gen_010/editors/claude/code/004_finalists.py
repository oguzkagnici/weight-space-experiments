# Hypothesis: many wrong finished answers are a wrong pick between two finalists, one of which is gold; then a better
# finalist decision is the lever. Detect "X or Y" finalist statements (letters) in the last 40% of each finished trace.
import json, re, collections
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
fin=[r for r in rows if not r['cut'] and r['pred']]
pat=re.compile(r"\(?\b([A-J])\)?\s*(?:or|vs\.?|versus)\s*\(?([A-J])\b\)?")
pat2=re.compile(r"(?:between|either)\s*\(?([A-J])\)?\s*(?:and|or)\s*\(?([A-J])\)?")
stats=collections.Counter()
for r in fin:
    t=r['think']; tail=t[int(len(t)*0.5):]
    pairs=[m.groups() for m in pat2.finditer(tail)]+[m.groups() for m in pat.finditer(tail) if m.group(1)!=m.group(2)]
    pairs=[p for p in pairs if p[0]!=p[1]]
    key=('right' if r['correct'] else 'wrong')
    if not pairs: stats[key,'no_pair']+=1; continue
    last=pairs[-1]
    g=r['gold'] in last; p=r['pred'] in last
    stats[key,'pair']+=1; stats[key,'gold_in_last_pair']+=g; stats[key,'pred_in_last_pair']+=p
for k in sorted(stats): print(k, stats[k])
