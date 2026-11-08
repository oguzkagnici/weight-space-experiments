# Hypothesis: the paragraph after a sweep's last option shows what the parent writes at sweep ends (look at examples).
import json, re
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
pat=re.compile(r'^\s*(?:[-*]\s*)?(?:\*\*)?(?:Option\s*)?\(?([A-J])\)[\s:.]|^\s*(?:[-*]\s*)?(?:\*\*)?Option\s*\(?([A-J])\)?\b',re.M)
for r in [B[2],B[5],B[4]]:
    t=r['think']; ms=list(pat.finditer(t))
    for m in ms:
        L=m.group(1) or m.group(2)
        if L=='J':
            # find end of this paragraph
            e=t.find('\n\n',m.end())
            print(r['id'],'...',repr(t[m.start():m.start()+120]),'\n  NEXT>>',repr(t[e:e+200]))
