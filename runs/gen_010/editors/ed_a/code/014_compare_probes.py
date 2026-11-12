# Hypothesis: the ramp-removal edit flattens the non-finalist letter ramp at decision probes and moves P toward the later finalist mainly where the pair is near-tied.
import json, numpy as np, sys
from wsp.editor import *
sel=json.load(open('work/probe_sel.json')); L='ABCDEFGHIJ'
E=load_original('model.embed_tokens.weight')[LETTER_IDS].numpy(); g=load_original('model.embedding_norm.weight').numpy()
def get(d):
    ti=np.load(d+'/topk_ids.npy'); tl=np.load(d+'/topk_logprobs.npy'); h=np.load(d+'/h24.npy')
    LL=(h/np.sqrt((h**2).mean(1,keepdims=True)+1e-5)*g)@E.T
    P=np.zeros((len(sel),10))
    for k in range(len(sel)):
        lp={int(i):float(v) for i,v in zip(ti[k],tl[k])}
        P[k]=[np.exp(lp.get(LETTER_IDS[j],-30)) for j in range(10)]
    return P,LL
P0,L0=get('work/fwd_01'); P1,L1=get(sys.argv[1])
def slope(LL):
    out=[]
    for k,it in enumerate(sel):
        m=[j for j in range(10) if L[j] not in (it['a'],it['b'],it['pred'] or 'x',it['gold'])]
        x=LL[k,m]-LL[k,m].mean(); out.append(np.polyfit(np.array(m),x,1)[0])
    return np.mean(out)
print('non-finalist slope orig %.3f edited %.3f'%(slope(L0),slope(L1)))
T=0.6
for nm,P in (('orig',P0),('edit',P1)):
    ge=[];gl=[]; dl=[]
    for k,it in enumerate(sel):
        a,b=sorted([it['a'],it['b']]); ia,ib=L.index(a),L.index(b)
        dl.append(np.log(P[k,ia]+1e-9)-np.log(P[k,ib]+1e-9))
        if it['gold']==a: ge.append(P[k,ia]**(1/T)/(P[k,ia]**(1/T)+P[k,ib]**(1/T)+1e-12))
        elif it['gold']==b: gl.append(P[k,ib]**(1/T)/(P[k,ia]**(1/T)+P[k,ib]**(1/T)+1e-12))
    print(nm,'mean T0.6 pair-P(gold): gold earlier %.3f (n%d)  gold later %.3f (n%d)  overall %.3f; median dl %.2f'%(np.mean(ge),len(ge),np.mean(gl),len(gl),np.mean(ge+gl),np.median(np.clip(dl,-15,15))))
d=np.abs(np.log(P1+1e-9)-np.log(P0+1e-9)); print('mean |dlogP| over letters with P0>0.01: %.3f'%d[P0>0.01].mean())
print('top letter changed in', int((P0.argmax(1)!=P1.argmax(1)).sum()), 'of', len(sel))
