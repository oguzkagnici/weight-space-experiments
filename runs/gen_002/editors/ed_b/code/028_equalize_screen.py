# Hypothesis: equalizing the L22 leader-suppressor bank (A/B/C weak, D-J strong) removes an early-letter advantage; lens-screen down (claude g1), up and to-mean variants on 512 DEV (pass 3) and the S1 parent rows.
from wsp.editor import *
import numpy as np, json
D='work/fwd_03/score_3_acts/original/'
ids=json.load(open(D+'index.json'))['ids']; qs=questions(); k=np.array([len(qs[i]['options']) for i in ids]); g=np.array([qs[i]['answer_index'] for i in ids])
E=np.load('work/E_letters.npy').astype(np.float64); wn=np.load('work/wn.npy').astype(np.float64)
H24=np.load(D+'h24.npy').astype(np.float64); rms24=np.sqrt((H24**2).mean(1)+1e-5)
C=np.load('work/p3_comp_dl.npy'); fin=C.sum(0)
W2=load('model.layers.22.feed_forward.w2.weight').double().numpy(); wpf=load('model.layers.22.post_ffn_norm.weight').double().numpy()
A=np.load(D+'mlp22.npy').astype(np.float64); M=A@W2.T; rm=np.sqrt((M**2).mean(1)+1e-5)
R=((W2*wpf[:,None]*wn[:,None]).T@E.T); R=R-R.mean(1,keepdims=True)
sup={0:[389],1:[6620,5859],2:[1812],3:[464],4:[7357],5:[6492],6:[3344],7:[397],8:[2786],9:[3310]}
eff={0:-0.76,1:-1.04,2:-1.73,3:-3.43,4:-4.13,5:-3.09,6:-3.56,7:-3.35,8:-3.97,9:-3.15}
def dla(j): return (A[:,j]/(rm*rms24))[:,None]*R[j][None,:]
DL={j:dla(j) for X in sup for j in sup[X]}
def ev(L):
    a=np.mean([np.argmax(L[i,:k[i]])==g[i] for i in range(len(g))])
    lp=np.mean([L[i,g[i]]-np.log(np.exp(L[i,:k[i]]-L[i,:k[i]].max()).sum())-L[i,:k[i]].max() for i in range(len(g))])
    pc=np.bincount([np.argmax(L[i,:k[i]]) for i in range(len(g))],minlength=10)
    return a,lp,pc
def variant(scales):
    L=fin.copy()
    for X,js in sup.items():
        for j in js: L+= (scales[X]-1)*DL[j]
    return L
tgt=np.mean([eff[X] for X in range(3,10)])
V={'base':{X:1 for X in sup},
   'down_CJ_0.5 (g1 claude)':{X:(1 if X<2 else 0.5) for X in sup},
   'up_ABC_to_mean':{X:(tgt/eff[X] if X<3 else 1) for X in sup},
   'all_to_mean':{X:tgt/eff[X] for X in sup},
   'all_to_-2':{X:-2/eff[X] for X in sup},
   'up_ABC_half':{X:(1+0.5*(tgt/eff[X]-1) if X<3 else 1) for X in sup}}
for nm,s in V.items():
    a,lp,pc=ev(variant(s)); print(f'{nm:26s} acc {a:.3f} logp {lp:.3f} preds {pc}', {('ABC'[X]):round(s[X],2) for X in range(3)})
print('gold', np.bincount(g,minlength=10))
