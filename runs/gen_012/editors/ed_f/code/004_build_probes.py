# Hypothesis H1 (see notes): build forced-answer probes after thinking prefixes of 8,192 bank traces (sample 0, 96 questions; + passk sample 1 as extra) for a read pass.
import json, collections, numpy as np, re, hashlib
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
tb=[json.loads(l) for l in open(S+'trace_bank_gen008champ_cap8192_det1.jsonl')]
pk=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
SUF="\n\nSo the answer is ("
def probe(row, frac, maxtok):
    full=trace_text(row); prompt=render_think(row['id'])
    assert full.startswith(prompt)
    gen=full[len(prompt):]
    th=gen.split('</think>')[0]
    # paragraph breaks
    br=[m.start() for m in re.finditer(r'\n\n',th)]
    if not br: return None
    target=frac*len(th)
    cands=[b for b in br if b<=target] or br[:1]
    b=cands[-1]
    while True:
        text=prompt+th[:b].rstrip()+SUF
        n=len(tok(text)['input_ids']) if isinstance(tok(text),dict) or hasattr(tok(text),'keys') else len(tok.encode(text))
        if n<=maxtok or len(cands)<=1: break
        cands=cands[:-1]; b=cands[-1]
    if n>maxtok: return None
    return dict(id=row['id'],sample=row.get('sample',0),gold=row['gold'],pred=row['pred'],correct=row['correct'],cut=row['cut'],
                frac=frac,pos=b/len(th),ntok=n,text=text,category=row['category'])
P=[]
for r in tb:
    p=probe(r,0.40,1900)
    if p: p['src']='bank40'; P.append(p)
tot=sum(p['ntok'] for p in P); print('bank40',len(P),tot)
# extra: passk sample 1 at 15% (different trajectory, early point), as many as fit
extra=[]
for r in pk:
    if r['sample'] not in (1,2): continue
    p=probe(r,0.15,1000)
    if p: p['src']='pk%d_15'%r['sample']; extra.append(p)
extra.sort(key=lambda p:p['sample'])
for p in extra:
    if tot+p['ntok']<=196000: P.append(p); tot+=p['ntok']
print('total probes',len(P),'tokens',tot, collections.Counter(p['src'] for p in P))
# split by question id hash
for p in P: p['split']='A' if int(hashlib.md5(p['id'].encode()).hexdigest(),16)%2==0 else 'B'
print(collections.Counter((p['src'],p['split']) for p in P))
print('has stated answer in prefix:',sum(bool(re.search(r'answer is \(?[A-J]\)?',p['text'][len(render_think(p['id'])):])) for p in P))
json.dump(P,open('work/probes.json','w'))
print(repr(P[0]['text'][-200:]))
ids=tok.encode(P[0]['text']); print(ids[-6:], [tok.decode([i]) for i in ids[-6:]]); print(LETTER_IDS)
req={"type":"read","model":"original","texts":[p['text'] for p in P],
     "return":{"topk":20,"hidden_states":list(range(25)),"mlp":list(range(0,24)),"positions":"last"}}
json.dump(req,open('work/req_p1.json','w'))
