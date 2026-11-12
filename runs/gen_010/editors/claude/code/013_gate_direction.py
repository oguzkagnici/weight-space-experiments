# Hypothesis: the L23 MLP input separates "paragraph break after a HEDGED commitment" from breaks after FIRM commitments
# and from other tokens (mean-difference direction; leave-one-text-out AUC). Also: what does the parent write next there?
import json, numpy as np, glob, collections
from wsp.editor import *
items=json.load(open('work/read4_index.json')); meta=json.load(open('work/fwd_04/meta.json'))
f=[p for p in glob.glob('work/fwd_04/*.npy') if 'w1' in p or 'inputs' in p]; print(f, meta['files'].keys())
X=np.load(f[0],mmap_mode='r'); print(X.shape, X.dtype)
TI=np.load('work/fwd_04/topk_ids.npy',mmap_mode='r'); TL=np.load('work/fwd_04/topk_logprobs.npy',mmap_mode='r')
offs=np.cumsum([0]+meta['n_tokens'])
H=[];F=[];O=[];grpH=[];grpF=[]
rng=np.random.default_rng(0)
for t,(it,o) in enumerate(zip(items,offs[:-1])):
    mk={ti:h for ti,h,_ in it['marks']}
    for ti,h in mk.items():
        (H if h else F).append(o+ti); (grpH if h else grpF).append(t)
    gen=np.arange(o+it['plen'],o+it['n']); gen=gen[~np.isin(gen,[o+k for k in mk])]
    O.extend(rng.choice(gen,size=min(150,len(gen)),replace=False))
X=np.asarray(X,dtype=np.float32) if X.shape[0]<200000 else X
xh,xf,xo=X[H],X[F],X[O]
mu=X[np.array(O)].mean(0)
print('hedged',len(H),'firm',len(F),'other sample',len(O))
from sklearn.metrics import roc_auc_score
gH=np.array(grpH); gF=np.array(grpF)
def loto(neg, gneg):
    sh=[];sn=[]
    for t in set(gH)|set(gneg):
        d=xh[gH!=t].mean(0)-np.concatenate([neg[gneg!=t]]).mean(0)
        sh+=list(xh[gH==t]@d); sn+=list(neg[gneg==t]@d)
    return roc_auc_score([1]*len(sh)+[0]*len(sn), sh+sn)
print('LOTO AUC hedged vs firm breaks:', round(loto(xf,gF),3))
gO=np.repeat(np.arange(len(items)),[min(150,it['n']-it['plen']-len(it['marks'])) for it in items])[:len(O)]
d=xh.mean(0)-0.5*xf.mean(0)-0.5*xo.mean(0)
s=lambda A: A@d
print('scores: hedged',np.percentile(s(xh),[10,50,90]).round(2),'firm',np.percentile(s(xf),[10,50,90]).round(2),'other',np.percentile(s(xo),[50,99,99.9]).round(2))
np.save('work/d_hedged.npy',d); np.save('work/mu_other.npy',mu)
# what comes next at hedged / firm breaks (top-1 tokens)
for name,P in (('hedged',H),('firm',F)):
    c=collections.Counter(tok.decode([int(TI[p][0])]).strip() or repr(tok.decode([int(TI[p][0])])) for p in P)
    print(name,'top-1 next:',c.most_common(10))
