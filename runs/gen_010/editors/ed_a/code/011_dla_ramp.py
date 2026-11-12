# Hypothesis: at forced-commit probes the earlier-finalist preference is (partly) a content-free letter ramp; measure the ramp in final letter logits and per-layer contributions (logit lens of layer deltas) to logit(earlier)-logit(later) and to the ramp.
import json, numpy as np
from wsp.editor import *
d='work/fwd_01'
sel=json.load(open('work/probe_sel.json')); L='ABCDEFGHIJ'
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy(); g=load('model.embedding_norm.weight').numpy()
H={l:np.load(f'{d}/h{l}.npy') for l in range(14,25)}
rms=np.sqrt((H[24]**2).mean(1,keepdims=True)+1e-5)
LL=(H[24]/rms*g)@E.T
r=np.arange(10)-4.5
# ramp in final letter logits over NON-finalist letters (content mostly absent)
resid=[]
for k,it in enumerate(sel):
    m=[j for j in range(10) if L[j] not in (it['a'],it['b'],it['pred'] or 'x',it['gold'])]
    x=LL[k,m]-LL[k,m].mean(); resid.append(np.polyfit(np.array(m),x,1)[0])
print('slope of non-finalist letter logits per letter step: mean %.3f sd %.3f'%(np.mean(resid),np.std(resid)))
# per-layer contribution to dl (earlier - later) and to ramp slope
dl=[];slopes={l:[] for l in range(15,25)}; cont={l:[] for l in range(15,25)}
for k,it in enumerate(sel):
    a,b=sorted([it['a'],it['b']]); ia,ib=L.index(a),L.index(b)
    for l in range(15,25):
        delta=(H[l][k]-H[l-1][k])/rms[k]*g
        v=delta@E.T
        cont[l].append(v[ia]-v[ib]); slopes[l].append(np.polyfit(np.arange(10),v-v.mean(),1)[0])
print('layer: mean contrib to logit(earlier)-logit(later) | mean ramp slope written')
for l in range(15,25): print(l, '%.2f'%np.mean(cont[l]), '%.3f'%np.mean(slopes[l]))
print('h14 lens dl mean', np.mean([((H[14][k]/np.sqrt((H[14][k]**2).mean()))*g@E.T)[L.index(min(it['a'],it['b']))]-((H[14][k]/np.sqrt((H[14][k]**2).mean()))*g@E.T)[L.index(max(it['a'],it['b']))] for k,it in enumerate(sel)]))
