# Hypothesis: cut traces express a leaning (a letter the model favours) early; if that leaning is often gold, a commit-when-leaning mechanism would pay off.
import json, re, numpy as np, collections
from wsp.editor import *
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')]
pat=re.compile(r"(answer is|answer:|likely|correct answer|best answer|best fit|best match|choose|pick|go with|lean(?:ing)? (?:toward|towards)|answer would be|answer should be|so the answer|thus the answer|it's|is)\s*(?:option\s*)?\(?\**([A-J])\b\)?", re.I)
pat=re.compile(r"(?:answer is|answer:|likely answer|likely|correct answer is|best answer|answer would be|answer should be|choose|pick|go with|lean(?:ing)? (?:toward|towards)|select)\s*(?:is\s*)?(?:option\s*)?\(([A-J])\)", re.I)
out=[]
for r in rows:
    t=r['think']; L=len(t)
    ms=[(m.start()/L, m.group(1)) for m in pat.finditer(t)]
    out.append((r['cut'],r['correct'],r['gold'],r['pred'],ms,r['tokens']))
for cut in (0,1):
    s=[o for o in out if o[0]==cut]
    has=[o for o in s if o[4]]
    fg=[o[4][0][1]==o[2] for o in has]; lg=[o[4][-1][1]==o[2] for o in has]
    anyg=[any(x[1]==o[2] for x in o[4]) for o in has]
    nl=[len(set(x[1] for x in o[4])) for o in has]
    print('cut',cut,'N',len(s),'with leaning',len(has),'first=gold',sum(fg),'last=gold',sum(lg),'any=gold',sum(anyg),'mean distinct letters %.2f'%np.mean(nl) if has else '', 'mean first pos %.2f'%np.mean([o[4][0][0] for o in has]))
    if cut==0: print('  finished: first leaning==pred',sum(o[4][0][1]==o[3] for o in has))
for o in out:
    if o[0]==1 and o[4]: print(o[2], [(round(p,2),l) for p,l in o[4]][:8])
