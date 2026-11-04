# Hypothesis: the L22 (and L23) letter-suppressor neurons, which fire when their letter is the retrieved candidate and push it down, are unequal across letters (weak for A/B) -> early-letter bias, and they also suppress correct answers. Direct-effect estimate of scaling them (no network simulation; final rms held fixed).
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions()
nopt=np.array([len(Q[r['id']]['options']) for r in rows]); ans=np.array([r['answer'] for r in rows]); m10=nopt==10
mask=np.arange(10)[None]<nopt[:,None]
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=np.where(mask,(h24/r)@U,-1e9)
top22=np.load('work/top22.npy')
fam={}
contribs={}
for l in [22,23]:
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5)
    Dn=(W2*pw[:,None]).T@U   # uncentered letter logit contributions
    Dc=Dn-Dn.mean(1,keepdims=True)
    k=np.abs(Dc).argmax(1); share=np.abs(Dc).max(1)/(np.abs(Dc).sum(1)+1e-9)
    s=(1/(rx*r))[:,0]
    for n in range(7680):
        if share[n]<0.45: continue
        c=a[:,n]*s*Dc[n,k[n]]
        sel=top22==k[n]
        if sel.sum()<3: continue
        if c[sel].mean()<-0.5 and c[sel].mean()<3*c[~sel].mean():
            fam[(l,n)]='ABCDEFGHIJ'[k[n]]
            contribs[(l,n)]=(a[:,n]*s)[:,None]*Dc[n][None]   # n x 10
print('family size',len(fam))
for key,v in sorted(fam.items(),key=lambda x:(x[0][0],x[1])):
    c=contribs[key]; kk='ABCDEFGHIJ'.index(v); sel=top22==kk
    print(key,v,'mean on top',round(c[sel,kk].mean(),2),'off',round(c[~sel,kk].mean(),2),'freq top',sel.sum())
def evaluate(Ladj):
    pred=Ladj.argmax(1); acc=(pred==ans).mean()
    lp=Ladj-np.log(np.exp(Ladj-Ladj.max(1,keepdims=True)).sum(1,keepdims=True))-Ladj.max(1,keepdims=True)
    return acc, lp[np.arange(len(ans)),ans].mean(), np.bincount(pred,minlength=10)
print('base',evaluate(L))
for layers in [(22,),(23,),(22,23)]:
    S=sum(v for k_,v in contribs.items() if k_[0] in layers)
    for alpha in [0.75,0.5,0.25,0.0]:
        acc,lp,cnt=evaluate(L-(1-alpha)*np.where(mask,S,0))
        print(layers,alpha,round(acc,3),round(lp,3),cnt)
