# Hypothesis: where the FINAL commitment is hedged and the thinking closes soon after it, the answer is often wrong; a
# re-check there (instead of the parent's wrap-up) could help. Look at the LAST commitment before </think>.
import json, re, collections, numpy as np
src=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
seen={(r['id'],r['sample']) for r in src}
src+=[r for r in map(json.loads,open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl')) if (r['id'],r.get('sample',0)) not in seen]
fin=[r for r in src if not r['cut'] and r['pred']]
COM=re.compile(r"(?:go with|answer is|answer:|answer =|choose|pick|select)\s*(?:option\s*)?\(?([A-J])\)?(?![a-z])",re.I)
HEDGE=re.compile(r"guess|likely|probably|maybe|perhaps|not sure|uncertain|seems|might|plausibl|lean",re.I)
st=collections.defaultdict(list)
for r in fin:
    t=r['think']; ms=list(COM.finditer(t))
    if not ms: continue
    # first commitment that has the final letter and after which no other letter is committed
    last=ms[-1]; j=len(ms)-1
    while j>0 and ms[j-1].group(1).upper()==last.group(1).upper(): j-=1
    m=ms[j]; ps=t.rfind('\n',0,m.start()); ctx=t[max(ps,m.start()-250):m.end()]
    h='hedged' if HEDGE.search(ctx) else 'firm'
    nfirst=sum(1 for x in ms if x.group(1).upper()!=m.group(1).upper())
    st[h].append((r['correct'], len(t)-m.end(), nfirst>0, len(t)))
for h,v in st.items():
    a=np.array(v,dtype=float)
    print(h,'n',len(v),'acc',round(100*a[:,0].mean(),1),'median chars after settling',int(np.median(a[:,1])),'had earlier other letter',int(a[:,2].sum()),'median trace chars',int(np.median(a[:,3])))
    short=a[a[:,1]<600]; print('   closes within 600 chars:',len(short),'acc',round(100*short[:,0].mean(),1) if len(short) else None)
