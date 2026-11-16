# Hypothesis (pass 3): (1b) attention from option-judging positions to the stem differs between right and wrong samples of the same question; plus the data for a position-gated L22 stem-attention edit (query inputs at judging vs background positions, key inputs at stem / option tokens).
import json, sys, re, importlib, collections, numpy as np, random
sys.path.insert(0,'code'); m=importlib.import_module('001_overview'); b=importlib.import_module('005_build_ins')
from wsp.editor import *
random.seed(14); Q=questions(); rows=m.load_rows(); ntok=b.ntok; QUANT={'business','chemistry','physics','math','engineering'}
meta0=json.load(open('work/p3_meta.json')); texts0=json.load(open('work/p3_texts.json'))
P=[(t,mt) for t,mt in zip(texts0,meta0) if not (mt['kind']=='ins' and mt['arm']=='INF')]
judge_q=sorted(set(mt['id'] for _,mt in P if mt['kind']=='judge'))
def prefix_at_token(text,k):
    e=tok(text,add_special_tokens=False,return_offsets_mapping=True); t=text[:e['offset_mapping'][k][1]]
    return t if tok(t,add_special_tokens=False)['input_ids']==e['input_ids'][:k+1] else None
# B: key texts (prompt cut at stem / option tokens)
keyq=sorted(set(judge_q)|set(b.ANN))
for q in keyq:
    Pm=render_think(q); e=tok(Pm,add_special_tokens=False,return_offsets_mapping=True); om=e['offset_mapping']
    qs=Pm.find(Q[q]['question']); qe=qs+len(Q[q]['question']); ins=Pm.find('Think step by step')
    stem=[i for i,(a,c) in enumerate(om) if qs<=a<qe]; opt=[i for i,(a,c) in enumerate(om) if qe<=a<ins]
    pick=set(random.sample(stem,min(10,len(stem))))
    cue=b.ANN[q][0] if q in b.ANN else None
    cuepos=[]
    if cue:
        cs=Pm.find(cue); cuepos=[i for i,(a,c) in enumerate(om) if cs<=a<cs+len(cue)]; pick|=set(random.sample(cuepos,min(3,len(cuepos))))
    for i in sorted(pick):
        t=prefix_at_token(Pm,i)
        if t: P.append((t,dict(kind='key',reg='cue' if i in cuepos else 'stem',id=q,pos=i)))
    for i in sorted(random.sample(opt,min(5,len(opt)))):
        t=prefix_at_token(Pm,i)
        if t: P.append((t,dict(kind='key',reg='opt',id=q,pos=i)))
# C: background trace positions and extra judging positions from other verbal questions
others=collections.defaultdict(list)
for r in rows:
    if r['id'] not in keyq and r['category'] not in QUANT: others[r['id']].append(r)
oq=sorted(others); random.shuffle(oq)
nb=nj=0
for q in oq:
    r=random.choice(others[q]); T,off,think,early,late=b.cuts(r)
    e=tok(T,add_special_tokens=False,return_offsets_mapping=True); om=e['offset_mapping']; t0=next(i for i,(a,c) in enumerate(om) if a>=off)
    if nb<40 and len(om)>t0+80:
        k=random.randint(t0+40,min(len(om)-2,t0+650)); t=prefix_at_token(T,k)
        if t: P.append((t,dict(kind='bg',id=q,sample=r['sample'],pos=k,tail=t[-40:]))); nb+=1
    if nj<40 and early is not None:
        mm=b.LAB.match(think[early:early+40]); line=think[early:early+220].split('\n')[0]; k=line.find(':'); en=early+(k+1 if 0<k<200 else mm.end())
        t=T[:off+en]
        if ntok(t)<=1500: P.append((t,dict(kind='judge2',id=q,sample=r['sample'],letter=mm.group(1),status='cut' if r['cut'] else ('right' if r['correct'] else 'wrong'),colon=int(0<k<200),tail=t[-60:]))); nj+=1
tot=sum(ntok(t) for t,_ in P); print('texts',len(P),'tokens',tot,'max',max(ntok(t) for t,_ in P),collections.Counter((mt['kind'],mt.get('arm'),mt.get('reg'),mt.get('status')) for _,mt in P))
assert tot<197000
json.dump([mt for _,mt in P],open('work/p3b_meta.json','w')); json.dump([t for t,_ in P],open('work/p3b_texts.json','w'))
mods=['model.layers.%d.self_attn.q_proj'%l for l in (10,14,18,22)]
json.dump({"type":"read","model":"original","texts":[t for t,_ in P],"return":{"token_logprobs":True,"topk":50,"positions":"last","attn_last":[2,4,6,8,10,12,14,16,18,20,22],"inputs":mods}},open('work/req_p3.json','w'))
asc=lambda t:t.encode('ascii','replace').decode()
for kd in ('key','bg','judge2','judge'):
    for t,mt in [x for x in P if x[1]['kind']==kd][:3]: print(kd,mt.get('id'),mt.get('reg'),'| tail:',repr(asc(t[-90:])))
