# Hypothesis: many cut traces state (hedged) answer letters mid-trace; measure how often those letters are right and where they occur.
import json, re, collections, numpy as np
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
PAT=re.compile(r"(?:answer|choice|option|pick|choose|select)\s*(?:is|would be|=|:)?\s*(?:likely|probably|maybe|perhaps|indeed|definitely)?\s*(?:option\s*)?\(([A-J])\)|(?:answer|choice)\s*(?:is|would be|:)\s*(?:likely\s*|probably\s*)?(?:option\s*)?([A-J])\b",re.I)
stats=collections.Counter()
for b in B:
    if not b['cut']: continue
    t=b['think']; L=len(t)
    hits=[(m.start(), (m.group(1) or m.group(2)).upper()) for m in PAT.finditer(t)]
    if not hits: stats['none']+=1; continue
    stats['any']+=1
    first=hits[0]; last=hits[-1]
    stats['first_ok']+= first[1]==b['gold']; stats['last_ok']+= last[1]==b['gold']
    stats['gold_ever']+= any(h[1]==b['gold'] for h in hits)
    print(b['id'], b['category'][:6], 'gold',b['gold'], 'n',len(hits), 'first',first[1],round(first[0]/L,2),'last',last[1],round(last[0]/L,2), ''.join(h[1] for h in hits)[:30])
print(stats)
