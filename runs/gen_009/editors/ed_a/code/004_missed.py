# Hypothesis: cut traces without a detected sweep use other option formats or are computation traces; inspect.
import json, re
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
S=json.load(open('work/sweeps.json'))
for r in B:
    if r['cut'] and not S[r['id']]:
        t=r['think']
        lines=[l for l in t.split('\n') if re.search(r'\b(?:option|Option)\s*[A-J]\b|^\s*[-*]?\s*\(?[A-J][):.]',l)]
        print(r['id'],r['category'],len(t),'optlines',len(lines),'|',' || '.join(l[:50] for l in lines[:4]).encode('ascii','replace').decode())
