# Hypothesis: L22/L23 MLPs contain letter-specific "anti-copy" neurons that suppress a retrieved option letter; they are stronger/more numerous for late letters (H,I,J) and cause the early-letter preference. Direct-logit attribution from recorded acts.
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions()
nopt=np.array([len(Q[r['id']]['options']) for r in rows]); ans=np.array([r['answer'] for r in rows]); m10=nopt==10
mask=np.arange(10)[None]<nopt[:,None]
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=(h24/r)@U
mix22=np.load(D+'out_model.layers.22.post_operator_norm.npy').astype(np.float32)
top22=np.where(mask,(mix22@Uc)/r,-1e9).argmax(1)
C={}
for l in [20,21,22,23]:
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5)
    Dn=(W2*pw[:,None]).T@Uc
    s=(1/(rx*r))[:,0]
    C[l]=(a,s,Dn)
    # letter specificity: dominant letter and its share
    k=np.abs(Dn).argmax(1); share=np.abs(Dn).max(1)/ (np.abs(Dn).sum(1)+1e-9)
    # signed per-question contribution to letter k
    contrib=(a*s[:,None])*Dn[np.arange(7680),k][None]  # n x 7680
    # how often is that neuron's letter the L22-retrieved one when it fires strongly
    print(f'== L{l}')
    sel=[]
    for n in np.argsort(-np.abs(contrib).mean(0))[:400]:
        if share[n]<0.35: continue
        c=contrib[:,n]; 
        on_top=c[top22==k[n]].mean() if (top22==k[n]).any() else 0
        off=c[top22!=k[n]].mean()
        on_ans=c[ans==k[n]].mean() if (ans==k[n]).any() else 0
        sel.append((n,'ABCDEFGHIJ'[k[n]],round(share[n],2),round(c.mean(),3),round(on_top,3),round(off,3),round(on_ans,3)))
    for s_ in sel[:14]: print('  ',s_)
    # per letter: total mean contribution of letter-specific (share>0.35) neurons, split by sign
    spec=share>0.35
    tot=np.zeros(10); neg=np.zeros(10)
    for kk in range(10):
        idx=np.where(spec&(k==kk))[0]
        cm=contrib[m10][:,idx].mean(0)
        tot[kk]=cm.sum(); neg[kk]=cm[cm<0].sum()
    print('  letter-specific total',np.round(tot,2)); print('  letter-specific neg  ',np.round(neg,2))
np.save('work/top22.npy',top22)
