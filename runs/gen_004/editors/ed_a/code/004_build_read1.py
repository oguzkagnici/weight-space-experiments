# Hypothesis: at the paragraph break right after a conclusion paragraph, the next-token choice (reconsider "But/Let's/Check" vs commit "Hence/Therefore") is set by identifiable late components; build a read pass with prefixes ending at those breaks (plus control breaks after non-conclusion paragraphs).
import json, re, random
from wsp.editor import *
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl')}
D=json.load(open('work/decisions_dev.json'))
split=json.load(open('work/split.json'))
random.seed(1)
concl=re.compile(r'^(Thus|So|Therefore|Hence)\b')
items=[]
for d in D:
    items.append({'kind':'concl','id':d['id'],'char':d['char'],'label':d['label'],'cut':d['cut']})
# controls: non-conclusion paragraph breaks in DEV traces, 1 per trace at a random paragraph in 30-90% of the trace
for qid in split['dev']:
    t=rows[qid]['think']; paras=t.split('\n\n'); pos=0; cands=[]
    for i,p in enumerate(paras[:-1]):
        pos+=len(p)+2
        if not concl.match(p.strip()) and 0.25<pos/len(t)<0.9 and len(p)>40:
            cands.append((pos,paras[i+1].split()[0] if paras[i+1].split() else ''))
    if cands:
        c=random.choice(cands)
        items.append({'kind':'ctrl','id':qid,'char':c[0],'label':'ctrl:'+c[1],'cut':rows[qid]['cut']})
texts=[];ntok=0;keep=[]
for it in items:
    r=rows[it['id']]
    txt=render_think(it['id'])+r['raw'][:it['char']]
    n=len(tok.encode(txt,add_special_tokens=False))
    last=tok.decode(tok.encode(txt,add_special_tokens=False)[-1:])
    it['ntok']=n; it['last']=last
    keep.append(it); texts.append(txt); ntok+=n
print(len(texts),'tokens',ntok, 'concl',sum(i['kind']=='concl' for i in keep))
import collections; print(collections.Counter(repr(i['last']) for i in keep).most_common(8))
# trim to 195k tokens by dropping longest controls / conclusion points from finished traces if needed
# priority to drop: controls first (longest first, keep >=20), then finished-trace reconsider points; never commits or cut-trace points
random.seed(2)
ctrl=[i for i in range(len(keep)) if keep[i]['kind']=='ctrl']; random.shuffle(ctrl)
fr=[i for i in range(len(keep)) if keep[i]['kind']=='concl' and keep[i]['label']=='reconsider' and keep[i]['cut']==0]; random.shuffle(fr)
order=ctrl[:-24]+fr
drop=set()
while ntok>195000:
    i=order.pop(0); drop.add(i); ntok-=keep[i]['ntok']
keep2=[keep[i] for i in range(len(keep)) if i not in drop]; texts2=[texts[i] for i in range(len(keep)) if i not in drop]
print('after trim',len(texts2),ntok, collections.Counter((i['kind'],i['label'] if i['kind']=='concl' else 'ctrl',i['cut']) for i in keep2))
json.dump(keep2,open('work/read1_items.json','w'))
attn=[2,4,6,8,10,12,14,16,18,20,22]
req={"type":"read","model":"original","texts":texts2,
     "return":{"topk":50,"hidden_states":list(range(25)),"mlp":list(range(24)),
               "inputs":[f"model.layers.{l}.self_attn.o_proj" for l in attn],
               "positions":"last","attn_last":attn}}
json.dump(req,open('work/req_read1.json','w'))
