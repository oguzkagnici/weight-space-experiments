# Hypothesis: the eventual (right or wrong) conclusion is fixed early; or the model's latent answer is right mid-trace and drifts.
# Forced-answer probes at 15/40/70% of the thinking (cut at the next newline), finished traces only, all passk questions;
# keep traces whose probes all fit in 4,090 tokens; prioritise mixed questions, then others; total <= 195k tokens.
import json, collections, random
from wsp.editor import *
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in rows: by[r['id']].append(r)
nc={q:sum(r['correct'] for r in v) for q,v in by.items()}
SUF="\n\nSo the answer is ("; FR=[0.15,0.40,0.70]
cand=[r for r in rows if not r['cut'] and r['pred']]
random.Random(0).shuffle(cand)
cand.sort(key=lambda r: 0 if 0<nc[r['id']]<4 else (1 if not r['correct'] else 2))   # mixed first, then wrong, then right
items=[]; texts=[]; total=0
for r in cand:
    t=r['think'][len('<think>'):] if r['think'].startswith('<think>') else r['think']
    pt=render_think(r['id']); add=[]; ok=True
    for f in FR:
        e=t.find('\n',int(len(t)*f)); e=len(t) if e<0 else e
        x=pt+t[:e]+SUF; n=len(tok(x,add_special_tokens=False).input_ids)
        if n>4090: ok=False; break
        add.append((f,x,n))
    if not ok: continue
    s=sum(a[2] for a in add)
    if total+s>195000: continue
    total+=s
    for f,x,n in add:
        texts.append(x); items.append({'id':r['id'],'sample':r['sample'],'gold':r['gold'],'pred':r['pred'],'correct':r['correct'],'frac':f,'mixed':0<nc[r['id']]<4,'nc':nc[r['id']]})
print(len(texts), total, 'traces', len(texts)//3, 'mixed traces', sum(i['mixed'] for i in items)//3, 'wrong traces', sum(not i['correct'] for i in items)//3)
json.dump(items,open('work/probe3_index.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"topk":50,"positions":"last"}},open('work/req_read3.json','w'))
