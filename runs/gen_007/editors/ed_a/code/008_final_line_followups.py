# Hypothesis: after a quoted final-output line '"The answer is (X)".' inside the think, the model sometimes continues re-checking instead of closing; count those and their outcomes.
import json, re
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
pat=re.compile(r'"The answer is \(([A-J])\)"\.?')
for r in B:
    th=r['think']; ms=list(pat.finditer(th))
    for m in ms:
        rest=th[m.end():]
        if len(rest.strip())>0:
            print(r['id'],'cut' if r['cut'] else 'fin','ok' if r['correct'] else 'wr',m.group(1),'gold',r['gold'],f"pos {m.start()/len(th):.2f} restchars {len(rest)}:",repr(rest[:90].encode('ascii','replace').decode()))
