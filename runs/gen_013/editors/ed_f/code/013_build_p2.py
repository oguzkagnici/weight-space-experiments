# Pass 2 (record): L23 MLP input at the pass-1 clean/big texts (to fit an error direction there) and at control positions (paragraph breaks, random tokens, prompt tokens) for the threshold.
import json, random, sys
sys.path.insert(0,'code')
import importlib; m=importlib.import_module('003_arith_check2')
from wsp.editor import *
random.seed(132)
meta=json.load(open('work/p1_meta.json')); T=json.load(open('work/p1_texts.json'))
keep=[i for i,x in enumerate(meta) if x['kind']!='small']
texts=[T[i] for i in keep]; M=[dict(meta[i],src='p1',p1=i) for i in keep]
ntok=lambda s:len(tok.encode(s,add_special_tokens=False))
rows=[r for r in m.load_rows() if r['sample']==0]; random.shuffle(rows)
budget=198000-sum(ntok(t) for t in texts); print('p1 part',len(texts),'tokens',198000-budget)
nA=nB=0
for r in rows:
    Tx=trace_text(r); st=Tx.index('<think>')+7; body=Tx[st:]
    enc=tok(Tx,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    n0=sum(1 for o in enc if o[1]<=st)
    # A: paragraph break within the first 650 thinking tokens
    brk=[i for i,o in enumerate(enc) if i>n0+20 and i<n0+650 and Tx[o[0]:o[1]].endswith('\n\n')]
    if brk and nA<70:
        i=random.choice(brk); t=Tx[:enc[i][1]]; c=i+1
        if budget-c>3000: texts.append(t); M.append(dict(src='ctlA',id=r['id'],cat=r['category'])); budget-=c; nA+=1
    if nB<55 and len(enc)>n0+60:
        i=random.randrange(n0+5,min(len(enc)-1,n0+650)); t=Tx[:enc[i][1]]; c=i+1
        if budget-c>3000: texts.append(t); M.append(dict(src='ctlB',id=r['id'],cat=r['category'])); budget-=c; nB+=1
r=rows[0]; Tx=trace_text(r); enc=tok(Tx,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
for k in (1,2,3,4,5,8,20,60):
    texts.append(Tx[:enc[k-1][1]]); M.append(dict(src='ctlC',k=k))
for r in rows[:6]:
    texts.append(render_think(r['id'])); M.append(dict(src='ctlC',k=-1))
print('texts',len(texts),'A',nA,'B',nB,'tokens',sum(ntok(t) for t in texts))
json.dump(M,open('work/p2_meta.json','w')); json.dump(texts,open('work/p2_texts.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"topk":50,"hidden_states":[22,23,24],"mlp":[23],"inputs":["model.layers.23.feed_forward","model.layers.22.feed_forward"],"positions":"last"}},open('work/req_p2.json','w'))
