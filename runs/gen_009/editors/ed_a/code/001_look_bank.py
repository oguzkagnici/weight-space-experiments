# Hypothesis: sweep ends (paragraph break after the last option of an option-by-option pass) can be found by text regex in bank traces.
import json, re, numpy as np
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
print(len(B), B[0].keys())
r=B[0]; print(r['id'],r['category'],r['cut'],r['tokens'])
t=r['think']
print(repr(t[:1500]))
# line-start patterns
pat=re.compile(r'^\s*(?:[-*]\s*)?(?:\*\*)?(?:Option\s*)?\(?([A-J])\)[\s:.]|^\s*(?:[-*]\s*)?(?:\*\*)?Option\s*\(?([A-J])\)?\b',re.M)
cnt=0
for r in B[:6]:
    ls=[ (m.group(1) or m.group(2)) for m in pat.finditer(r['think'])]
    print(r['id'],r['cut'],len(r['think']),''.join(ls)[:200])
