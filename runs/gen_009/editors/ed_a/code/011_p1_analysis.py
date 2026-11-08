# Hypothesis: (1) the parent rarely writes "Hence" at eval sweep ends; (2) after a forced "Hence" it often writes " answer"; (3) the forced letter is often right in cut traces.
import json, numpy as np, collections
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); D=json.load(open('work/pass1_plan.json'))
TI=np.load('work/fwd_01/topk_ids.npy',mmap_mode='r'); TL=np.load('work/fwd_01/topk_logprobs.npy',mmap_mode='r')
W=lambda s: tok.encode(s,add_special_tokens=False)
names=['Hence','Thus','We','Given','Option','Let','But','So','Therefore','The','Now','Between','Which','Alternatively','However','Since','Based','In','Looking','Answer','Final','I']
ids={n:W(n)[0] for n in names}
def dist(r):
    return {int(i):float(np.exp(l)) for i,l in zip(TI[r],TL[r])}
# (1) at eval sweep-end breaks
for kind in ['swend','lisend','brk']:
    sel=[r for r in rows if r[3]==kind]
    acc=collections.defaultdict(float)
    for r in sel:
        d=dist(r[0])
        for n,i in ids.items(): acc[n]+=d.get(i,0)
    print(kind,len(sel),' '.join('%s %.3f'%(n,acc[n]/max(1,len(sel))) for n in sorted(acc,key=lambda n:-acc[n])[:12]))
# (2) forced chain
for j,lab in [(1,'after Hence'),(2,'after answer'),(3,'after is')]:
    sel=[r for r in rows if r[3]=='forced%d'%j]
    acc=collections.Counter()
    for r in sel:
        for i,p in dist(r[0]).items(): acc[i]+=p
    print(lab,len(sel),[(tok.decode([i]).encode('ascii','replace').decode(),round(v/len(sel),3)) for i,v in acc.most_common(10)])
# (3) letters after "("
sel=[r for r in rows if r[3]=='forced4']
LI=LETTER_IDS
res=[]
for r in sel:
    ti=r[1]; m=D['meta'][ti]; d=dist(r[0]); q=questions()[m['id']]
    pl=np.array([d.get(i,0) for i in LI[:len(q['options'])]])
    res.append(dict(id=m['id'],cut=m['cut'],correct=m['correct'],which=m['which'],pg=pl[q['answer_index']]/max(pl.sum(),1e-9),arg=int(pl.argmax()==q['answer_index']),mass=pl.sum()))
for cut in [0,1]:
    for wh in ['first','last']:
        s=[x for x in res if x['cut']==cut and x['which']==wh]
        if s: print('cut',cut,wh,len(s),'argmax right %.2f  P(gold) %.2f  parent correct %.2f letter mass %.2f'%(np.mean([x['arg'] for x in s]),np.mean([x['pg'] for x in s]),np.mean([x['correct'] for x in s]),np.mean([x['mass'] for x in s])))
json.dump(res,open('work/p1_forced_letters.json','w'))
