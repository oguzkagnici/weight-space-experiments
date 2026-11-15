# Hypothesis (pass 1): the parent registers a corrupted step (arithmetic result F1; number mis-copied from the question F2) in its residual stream and/or by raising re-check openers at the next paragraph start.
import json, collections, re, sys, random
sys.path.insert(0,'code')
import importlib; m=importlib.import_module('003_arith_check2')
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
random.seed(13)
rows=m.load_rows()
for l in open(S+'trace_bank_gen008champ_cap3072_det1.jsonl'):
    r=json.loads(l); r['sample']=10; rows.append(r)       # sample 10 = the 3,072 bank trace (same parent, same seed as sample 0 -> same text up to the cap)
Q=questions()
ntok=lambda s:len(tok.encode(s,add_special_tokens=False))
def base(r):
    T=trace_text(r); raw=r['raw']; raw=raw[len('<think>'):] if raw.startswith('<think>') else raw
    assert T.endswith(raw); return T,len(T)-len(raw),raw
def sigpos(c):
    return [i for i,ch in enumerate(c) if ch.isdigit()]
def corrupt(c,kind):
    ds=sigpos(c); lead=next(i for i in ds if c[i]!='0') if any(c[i]!='0' for i in ds) else ds[0]
    sig=[i for i in ds if i>=lead]
    if 'e' in c.lower(): sig=[i for i in sig if i<c.lower().index('e')]
    i=sig[0] if kind=='big' else sig[-1]
    d=int(c[i]); choices=[x for x in ([d+1,d+2,d-1,d-2] if kind=='small' else [d+1,d+2,d+3,d-1,d-2,d-3]) if 0<=x<=9 and not (i==sig[0] and x==0)]
    return c[:i]+str(random.choice(choices))+c[i+1:]
eqs=m.scan(rows)
F1=[];perq=collections.Counter();pert=collections.Counter()
random.shuffle(eqs)
info={(r['id'],r['sample']):r for r in rows}
seen_txt=set()
for e in sorted(eqs,key=lambda e:e['cstart']):
    if not e['ok']: continue
    v=e['true']; C=m.f(e['c'])
    if abs(C-v)>1e-9*max(1,abs(v)): continue                       # exact results only
    digs=[ch for ch in e['c'] if ch.isdigit()]
    if len(''.join(digs).strip('0'))<2 or len(digs)>9: continue
    if m.f(e['a']) in (0,1) or m.f(e['b']) in (0,1): continue
    r=info[(e['id'],e['sample'])]; T,off,raw=base(r)
    assert raw[e['cstart']-7+0:e['cend']-7]==e['c'] if r['raw'].startswith('<think>') else True
    cs,ce=off+e['cstart']-7,off+e['cend']-7
    pre=T[:cs]
    key=(e['id'],pre[-60:],e['c'])
    if key in seen_txt: continue
    if perq[e["id"]]>=6 or pert[(e["id"],e["sample"])]>=3: continue
    if e['c'] in pre[off:] or e['c'].replace(',','') in T[:off].replace(',',''): continue   # result not seen before (trace or prompt/options)
    n=ntok(pre)
    if n>1500: continue
    seen_txt.add(key); perq[e['id']]+=1; pert[(e['id'],e['sample'])]+=1
    item=dict(fam='F1',id=e['id'],sample=e['sample'],cat=r['category'],expr=e['expr'],c=e['c'],ntok=n)
    for kind in ('clean','big','small'):
        c2=e['c'] if kind=='clean' else corrupt(e['c'],kind)
        item[kind]=pre+c2+'.\n\n'; item['c_'+kind]=c2
    F1.append(item)
print('F1 candidates',len(F1),'questions',len(perq),collections.Counter(i['cat'] for i in F1))
random.shuffle(F1)
# balance: at most 66, spread over questions
F1=sorted(F1,key=lambda i:i['ntok'])[:200]; random.shuffle(F1); F1=F1[:80]
# F2: restatement paragraph with a number copied from the question
F2=[];usedq=set()
NUMRE=re.compile(r'(?<![\w.,])\d+(?:,\d{3})*(?:\.\d+)?(?![\w]|[.,]\d)')
for r in rows:
    if r['id'] in usedq or r["sample"] not in (0,1,10): continue
    T,off,raw=base(r); qtext=Q[r['id']]['question']
    end=raw.find('\n\n')
    if end<40 or end>900: continue
    para=raw[:end]
    cands=[mm for mm in NUMRE.finditer(para) if re.search(r'(?<![\d.,])'+re.escape(mm.group(0))+r'(?![\d]|[.,]\d)',qtext) and len(mm.group(0).replace(',','').replace('.',''))>=2 and para.count(mm.group(0))==1]
    if not cands: continue
    mm=random.choice(cands); c=mm.group(0); c2=corrupt(c,'big')
    if re.search(r'(?<![\d.,])'+re.escape(c2)+r'(?![\d]|[.,]\d)',T[:off]): continue
    clean=T[:off]+para+'\n\n'; cor=T[:off]+para[:mm.start()]+c2+para[mm.end():]+'\n\n'
    usedq.add(r['id'])
    F2.append(dict(fam='F2',id=r['id'],sample=r['sample'],cat=r['category'],c=c,c_big=c2,clean=clean,big=cor,ntok=ntok(clean),ctx=para[max(0,mm.start()-50):mm.end()+30]))
print('F2 candidates',len(F2),collections.Counter(i['cat'] for i in F2))
random.shuffle(F2); F2=F2[:44]
texts=[];meta=[]
for it in F1:
    for kind in ('clean','big','small'):
        texts.append(it[kind]); meta.append(dict(fam='F1',kind=kind,id=it['id'],sample=it['sample'],cat=it['cat'],expr=it['expr'],c=it['c'],cv=it['c_'+kind],plen=len(it[kind])-len(it['c_'+kind])-3))
for it in F2:
    for kind in ('clean','big'):
        texts.append(it[kind]); meta.append(dict(fam='F2',kind=kind,id=it['id'],sample=it['sample'],cat=it['cat'],c=it['c'],cv=it['c'] if kind=='clean' else it['c_big'],ctx=it['ctx']))
tot=sum(ntok(t) for t in texts); print('texts',len(texts),'tokens',tot,'max',max(ntok(t) for t in texts))
json.dump(meta,open('work/p1_meta.json','w')); json.dump(texts,open('work/p1_texts.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"token_logprobs":True,"topk":50,"hidden_states":list(range(0,25)),"mlp":[18,19,20,21,22,23],"positions":"last"}},open('work/req_p1.json','w'))
asc=lambda t:t.encode('ascii','replace').decode()
for it in F1[:8]: print('F1',it['id'],it['cat'],it['ntok'],asc(it['expr']),'| big',it['c_big'],'small',it['c_small'],'| tail:',repr(asc(it['big'][-70:])))
for it in F2[:8]: print('F2',it['id'],it['cat'],it['ntok'],it['c'],'->',it['c_big'],'|',repr(asc(it['ctx'])))
