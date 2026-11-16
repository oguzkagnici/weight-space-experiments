# Hypothesis (passes 1-2): cue quote vs non-cue quote vs cue+implication insertions in wrong samples (late and early cut); attention from judging positions to the stem, right vs wrong samples.
import json, sys, re, importlib, collections, numpy as np
sys.path.insert(0,'code'); m=importlib.import_module('001_overview'); b=importlib.import_module('005_build_ins')
from wsp.editor import *
Q=questions(); rows=m.load_rows(); ANN=b.ANN; ntok=b.ntok
QUANT={'business','chemistry','physics','math','engineering'}
def ins(q,arm):
    c,n,i=ANN[q]
    if arm=='BASE': return ''
    if arm=='CUE': return b.TPL%c+'\n\n'
    if arm=='NON': return (b.TPL%n if n else b.TPLF%b.FILL)+'\n\n'
    if arm=='INF': return b.TPL%c+' '+i+'\n\n'
def fit(T,off,think,cut,maxtok=3850):
    if ntok(T[:off+cut])<=maxtok: return cut,0
    br=[mm.end() for mm in re.finditer(r'\n\n',think) if mm.end()<cut]
    for c in reversed(br):
        if ntok(T[:off+c])<=maxtok: return c,1
P1=[];P2=[]
def add(P,text,**meta): P.append((text,meta))
byq=collections.defaultdict(list)
for r in rows: byq[r['id']].append(r)
for r in rows:
    if r['id'] not in ANN: continue
    T,off,think,early,late=b.cuts(r); st='cut' if r['cut'] else ('right' if r['correct'] else 'wrong')
    wp=[x['pred'] for x in byq[r['id']] if not x['cut'] and not x['correct'] and x['pred']]
    modal=collections.Counter(wp).most_common(1)[0][0]
    meta=dict(id=r['id'],sample=r['sample'],status=st,gold=r['gold'],pred=r['pred'],wrong=r['pred'] if st=='wrong' else modal)
    if st=='wrong' and late:
        c,tr=fit(T,off,think,late)
        for arm in ('NON','CUE'): add(P1,T[:off+c]+ins(r['id'],arm)+b.PROBE,kind='ins',cut='late',arm=arm,trunc=tr,frac=c/len(think),**meta)
        add(P2,T[:off+c]+ins(r['id'],'INF')+b.PROBE,kind='ins',cut='late',arm='INF',trunc=tr,frac=c/len(think),**meta)
        add(P2,T[:off+c]+b.PROBE,kind='ins',cut='late',arm='BASE',trunc=tr,frac=c/len(think),prio=ntok(T[:off+c]),**meta)
    if early:
        for arm in ('NON','CUE','INF'):
            add(P1 if (st=='wrong' and arm!='INF') else P2,T[:off+early]+ins(r['id'],arm)+b.PROBE,kind='ins',cut='early',arm=arm,trunc=0,frac=early/len(think),**meta)
# step 1(b): judging positions in finished samples of mixed verbal questions and of cue questions with right samples
mixed=[q for q,rs in byq.items() if any(r['correct'] for r in rs) and any((not r['cut']) and (not r['correct']) for r in rs) and rs[0]['category'] not in QUANT]
for q in mixed:
    rs=byq[q]; wp=[x['pred'] for x in rs if not x['cut'] and not x['correct'] and x['pred']]; modal=collections.Counter(wp).most_common(1)[0][0]
    for r in rs:
        if r['cut']: continue
        T,off,think,early,late=b.cuts(r)
        if early is None: continue
        starts=[early]+[mm.end() for mm in re.finditer(r'\n\n',think) if mm.end()>early]
        for tgt,X in (('gold',r['gold']),('wrong',r['pred'] if not r['correct'] else modal)):
            for s in starts:
                mm=b.LAB.match(think[s:s+40])
                if mm and mm.group(1)==X:
                    line=think[s:s+220].split('\n')[0]; k=line.find(':')
                    e=s+(k+1 if 0<k<200 else mm.end())
                    if ntok(T[:off+e])<=3900: add(P2,T[:off+e],kind='judge',tgt=tgt,letter=X,id=q,sample=r['sample'],status='right' if r['correct'] else 'wrong',gold=r['gold'],pred=r['pred'],colon=int(0<k<200),frac=e/len(think),tail=think[max(0,e-60):e])
                    break
for name,P in (('p1',P1),('p2',P2)):
    tot=sum(ntok(t) for t,_ in P)
    if name=='p2' and tot>198000:                      # drop the longest late BASE texts until the pass fits
        base=sorted([i for i,(t,mt) in enumerate(P) if mt.get('arm')=='BASE'],key=lambda i:-P[i][1]['prio'])
        drop=set()
        for i in base:
            if tot<=198000: break
            tot-=ntok(P[i][0]); drop.add(i)
        P=[x for i,x in enumerate(P) if i not in drop]; print('dropped late BASE texts',len(drop))
    print(name,'texts',len(P),'tokens',tot,'max',max(ntok(t) for t,_ in P),collections.Counter((mt['kind'],mt.get('cut'),mt.get('arm'),mt.get('status')) for _,mt in P))
    json.dump([mt for _,mt in P],open('work/%s_meta.json'%name,'w')); json.dump([t for t,_ in P],open('work/%s_texts.json'%name,'w'))
    json.dump({"type":"read","model":"original","texts":[t for t,_ in P],"return":{"token_logprobs":True,"topk":50,"positions":"last","attn_last":[2,4,6,8,10,12,14,16,18,20,22]}},open('work/req_%s.json'%name,'w'))
asc=lambda t:t.encode('ascii','replace').decode()
for t,mt in P1[:4]+[x for x in P2 if x[1]['kind']=='judge'][:6]+[x for x in P2 if x[1].get('arm')=='INF'][:2]: print(mt.get('id'),mt.get('arm'),mt.get('cut'),mt.get('tgt'),'| tail:',repr(asc(t[-260:])))
