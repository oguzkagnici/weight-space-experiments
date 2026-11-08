# Hypothesis: some paragraph openers ("Thus", "Therefore", "Hence") lead to an answer statement in the same paragraph; pick the best commit opener.
import json, re, collections
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
ANS=re.compile(r"answer (?:is|would be|:)|answer:\s*\(?[A-J]\)?|\boption [A-J]\b is (?:the )?(?:correct|best|most)|correct (?:answer|option) is",re.I)
c=collections.Counter(); a=collections.Counter()
for b in B:
    for p in b['think'].split('\n\n'):
        p=p.strip()
        if not p: continue
        w=p.split()[0].strip(',:')
        c[w]+=1; a[w]+=bool(ANS.search(p[:200]))
for w,n in c.most_common(40): print(w, n, a[w], round(a[w]/n,2))
