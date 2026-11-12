# Pass 4 builder. Record the L23 MLP input (input of feed_forward.w1 = ffn_norm output) at ALL positions, plus top-10 next
# tokens, along finished traces that contain commitment statements, to build a gate for "paragraph break right after a
# HEDGED commitment" (vs firm commitments and all other tokens). Traces truncated to 4,090 tokens; total <= 150k tokens.
import json, re, random
from wsp.editor import *
src=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
seen={(r['id'],r['sample']) for r in src}
src+=[r for r in map(json.loads,open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl')) if (r['id'],r.get('sample',0)) not in seen]
COM=re.compile(r"(?:go with|answer is|answer:|answer =|choose|pick|select)\s*(?:option\s*)?\(?([A-J])\)?(?![a-z])",re.I)
HEDGE=re.compile(r"guess|likely|probably|maybe|perhaps|not sure|uncertain|seems|might|plausibl|lean",re.I)
random.Random(1).shuffle(src)
HP=re.compile(r'(?:guess|likely|probably|maybe|perhaps|not sure|seems|might|lean)[^\n]{0,120}(?:go with|answer is|choose|pick)',re.I)
src.sort(key=lambda r: 0 if HP.search(r['think']) else 1)   # traces with hedged commitments first
items=[]; texts=[]; total=0
for r in src:
    if r['cut'] or not r['pred']: continue
    pt=render_think(r['id']); body=r['raw'][len('<think>'):] if r['raw'].startswith('<think>') else r['raw']
    enc=tok(pt+body,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc.input_ids[:4090]; offs=enc.offset_mapping[:4090]
    if not ids: continue
    endc=offs[-1][1]; full=pt+body
    marks=[]
    for m in COM.finditer(full,len(pt)):
        nl=full.find('\n',m.end())
        if nl<0 or nl>=endc: continue
        ps=full.rfind('\n',0,m.start()); ctx=full[max(ps,m.start()-250):m.end()]
        h=1 if HEDGE.search(ctx) else 0
        ti=next((k for k,(a,b) in enumerate(offs) if a<=nl<b),None)
        if ti is not None: marks.append((ti,h,m.group(1).upper()))
    if not marks: continue
    hed=sum(h for _,h,_ in marks)
    if total+len(ids)>150000: continue
    total+=len(ids)
    texts.append(tok.decode(ids)); items.append({'id':r['id'],'sample':r.get('sample',0),'correct':r['correct'],'n':len(ids),'marks':marks,'plen':len(tok(pt,add_special_tokens=False).input_ids)})
print(len(texts), total, 'marks', sum(len(i['marks']) for i in items), 'hedged', sum(h for i in items for _,h,_ in i['marks']))
# verify re-tokenisation
bad=sum(len(tok(t,add_special_tokens=False).input_ids)!=i['n'] for t,i in zip(texts,items)); print('retok mismatches', bad)
json.dump(items,open('work/read4_index.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"topk":10,"positions":"all","inputs":["model.layers.23.feed_forward.w1"]}},open('work/req_read4.json','w'))
