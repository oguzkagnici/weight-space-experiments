# Hypothesis: (pass 3 analysis) targets get P~1; thinking-token |dlogp| is tiny (fold residual) and no gate fires out of sample.
import json, numpy as np
from wsp.editor import *
d=json.load(open('work/p3_texts.json')); D='work/fwd_03'
lp=np.load(f'{D}/token_logprobs.npy'); off=np.load(f'{D}/offsets.npy'); ti=np.load(f'{D}/topk_ids.npy'); tv=np.load(f'{D}/topk_logprobs.npy')
lp0=np.load('work/fwd_02/token_logprobs.npy')
TGT={14225,5141,124900}
n2=52; res={'qthe':[],'ans':[],'im':[]}; dl=[]
fires=[]
for i,t in enumerate(d['texts']):
    ids=tok(t,add_special_tokens=False).input_ids; b=off[i]-i; n=len(ids)
    L=lp[b:b+n-1]
    if i<n2:
        L0=lp0[b:b+n-1]
        if d['kind'][i]!='cut':
            p=ids.index(124902)
            res['qthe'].append(L[p]); res['ans'].append(L[p+1]); res['im'].append(L[p+6])
            dl.append(np.abs(L[:p]-L0[:p]))     # thinking + prompt part
        else: dl.append(np.abs(L-L0))
    # firing check on thinking tokens: top-1 is a target token but the actual next token is not
    end=ids.index(124902) if 124902 in ids else n-1
    for q in range(end):
        if int(ti[off[i]+q][0]) in TGT and ids[q+1]!=int(ti[off[i]+q][0]) and tv[off[i]+q][0]>-0.7:
            fires.append((d['ids'][i],q,tok.decode([ids[q]]).encode('ascii','replace').decode(),int(ti[off[i]+q][0]),round(float(tv[off[i]+q][0]),2)))
for k,v in res.items(): v=np.array(v); print(k,'P median',round(float(np.exp(np.median(v))),4),'min',round(float(np.exp(v.min())),4),'n<0.9',int((np.exp(v)<0.9).sum()))
dl=np.concatenate(dl); print('thinking |dlogp| mean',dl.mean(),'p99',np.quantile(dl,.99),'max',dl.max(),'n>0.01',(dl>0.01).sum(),'of',len(dl))
print('suspected firings in thinking:',len(fires),fires[:10])
