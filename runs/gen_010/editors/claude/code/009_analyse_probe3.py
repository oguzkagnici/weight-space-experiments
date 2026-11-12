# Pass 3: when is the conclusion fixed, and is the latent answer ever right mid-trace and then lost (drift)?
import json, numpy as np, collections
from wsp.editor import *
items=json.load(open('work/probe3_index.json'))
ids=np.load('work/fwd_03/topk_ids.npy'); lps=np.load('work/fwd_03/topk_logprobs.npy')
L="ABCDEFGHIJ"; inv={LETTER_IDS[k]:l for k,l in enumerate(L)}
for it,ti,tl in zip(items,ids,lps):
    pr=collections.Counter()
    for a,b in zip(ti,tl):
        if int(a) in inv: pr[inv[int(a)]]+=float(np.exp(b))
    s=sum(pr.values()); it['top']=max(pr,key=pr.get) if pr else None; it['ptop']=pr[it['top']]/s if s else 0; it['pgold']=pr[it['gold']]/s if s else 0
tr=collections.defaultdict(dict)
for it in items: tr[(it['id'],it['sample'])][it['frac']]=it
F=[0.15,0.40,0.70]
for f in F:
    v=[t[f] for t in tr.values()]
    print(f"frac {f}: latent==final {sum(x['top']==x['pred'] for x in v)}/{len(v)}, latent==gold {sum(x['top']==x['gold'] for x in v)}, "
          f"median p_top {np.median([x['ptop'] for x in v]):.2f}, mean p_gold {np.mean([x['pgold'] for x in v]):.2f}")
print('final correct', sum(t[F[0]]['correct'] for t in tr.values()), 'of', len(tr))
wr=[t for t in tr.values() if not t[F[0]]['correct']]
print('wrong traces:', len(wr), 'latent gold at some checkpoint:', sum(any(t[f]['top']==t[f]['gold'] for f in F) for t in wr),
      '| mean p_gold max over checkpoints', round(float(np.mean([max(t[f]['pgold'] for f in F) for t in wr])),2))
rt=[t for t in tr.values() if t[F[0]]['correct']]
print('right traces:', len(rt), 'latent wrong at 70%:', sum(t[0.70]['top']!=t[0.70]['gold'] for t in rt))
for t in wr: print(t[F[0]]['id'][-6:],t[F[0]]['sample'],'gold',t[F[0]]['gold'],'pred',t[F[0]]['pred'],[(t[f]['top'],round(t[f]['pgold'],2)) for f in F])
