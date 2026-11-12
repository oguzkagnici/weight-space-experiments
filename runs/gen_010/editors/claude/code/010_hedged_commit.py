# Hypothesis: a commitment stated with a hedge ("I'd guess (X)", "likely (X)") is often wrong, and when the model re-checks
# after a commitment it switches to the right answer more often than away from it. If so, re-checking specifically after
# HEDGED commitments (the opposite of the parent's L20 wrap-up, which stops re-checking after any stated answer) could be a
# content gain. Text statistics on all finished 8,192 traces (passk + trace bank).
import json, re, collections
src=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
seen={(r['id'],r['sample']) for r in src}
src+=[r for r in map(json.loads,open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl')) if (r['id'],r.get('sample',0)) not in seen]
fin=[r for r in src if not r['cut'] and r['pred']]
COM=re.compile(r"(?:go with|answer is|answer:|answer =|choose|pick|select)\s*(?:option\s*)?\(?([A-J])\)?(?![a-z])",re.I)
HEDGE=re.compile(r"guess|likely|probably|maybe|perhaps|not sure|uncertain|seems|might|plausibl|lean",re.I)
st=collections.Counter()
for r in fin:
    t=r['think']; ms=list(COM.finditer(t))
    if not ms: st['no_commit']+=1; continue
    m=ms[0]; par_start=t.rfind('\n',0,m.start())
    ctx=t[max(par_start,m.start()-250):m.end()]
    h='hedged' if HEDGE.search(ctx) else 'firm'
    first=m.group(1).upper(); final=r['pred']
    st[h,'n']+=1; st[h,'first_right']+=first==r['gold']; st[h,'final_right']+=final==r['gold']
    st[h,'switched']+=first!=final
    rest=t[m.end():]; st[h,'tail_chars']+=len(rest)
    st[h,'recheck_words']+=bool(re.match(r"\s*(?:\n+)?\s*(But|Wait|However|Let's double|Let us double|Check|Actually)",rest))
for k in sorted(st): print(k, st[k])
