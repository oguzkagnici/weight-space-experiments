# How often did a hedged commitment get followed by a re-check opener, parent vs edited, on the holdout48 traces?
import json, re, collections
COM=re.compile(r"(?:go with|answer is|answer:|answer =|choose|pick|select)\s*(?:option\s*)?\(?([A-J])\)?(?![a-z])",re.I)
HEDGE=re.compile(r"guess|likely|probably|maybe|perhaps|not sure|uncertain|seems|might|plausibl|lean",re.I)
RE=re.compile(r"\s*(But|Wait|Let|Alternatively|However|Hmm|Actually)\b")
def count(rows):
    c=collections.Counter()
    for r in rows:
        t=r['think']
        for m in COM.finditer(t):
            nl=t.find('\n',m.end())
            if nl<0: continue
            ps=t.rfind('\n',0,m.start()); h=bool(HEDGE.search(t[max(ps,m.start()-250):m.end()]))
            nxt=t[nl:].lstrip('\n')
            c['hedged' if h else 'firm']+=1; c[('hedged' if h else 'firm')+'_reopen']+=bool(RE.match(nxt))
    return dict(c)
par=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/holdout48_bank_gen008champ_cap8192_det1.jsonl')]
ed=[json.loads(l) for p in ('work/fwd_06/gen_6_edited.jsonl','work/fwd_07/gen_7_edited.jsonl') for l in open(p)]
print('parent',count(par)); print('edited',count(ed))
