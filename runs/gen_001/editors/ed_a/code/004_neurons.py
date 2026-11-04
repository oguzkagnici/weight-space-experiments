# Hypothesis: the early-letter bias is written by a small set of late MLP neurons (and/or heads) with roughly constant activation at the answer position; L22.mlp's negative "content" is suppression of the attention-retrieved letter.
from wsp.editor import *
import numpy as np, json, torch
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions()
nopt=np.array([len(Q[r['id']]['options']) for r in rows]); ans=np.array([r['answer'] for r in rows]); m10=nopt==10
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=np.where(np.arange(10)[None]<nopt[:,None],(h24/r)@U,-1e9); Lc10=L[m10]-L[m10].mean(1,keepdims=True)
b=Lc10.mean(0); bn=b/np.linalg.norm(b)
print('global bias',np.round(b,2))
# letter retrieved by L22 attention
mix22=np.load(D+'out_model.layers.22.post_operator_norm.npy').astype(np.float32)
c22=(mix22@Uc)/r; top22=np.where(np.arange(10)[None]<nopt[:,None],c22,-1e9).argmax(1)
print('L22.mix argmax == answer:',(top22==ans).mean())
allres=[]
for l in range(16,24):
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)  # n x 7680
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5)
    y=np.load(D+f'out_model.layers.{l}.post_ffn_norm.npy').astype(np.float32)
    err=np.abs(x/rx*pw-y).max()
    Dn=(W2*pw[:,None]).T@Uc   # 7680 x10 : per unit act contribution to centered letter logits (before scale)
    s=(1/(rx*r))[:,0]          # per question scale
    # bias of neuron n: mean_q a_qn s_q * Dn
    am=(a[m10]*s[m10,None]).mean(0)       # 7680
    biasn=am[:,None]*Dn                    # 7680x10
    proj=biasn@bn
    # content: contribution to correct letter, and to L22-retrieved letter
    cc=((a*s[:,None])[:,:,None]*Dn[None]) if False else None
    contr_ans=((a*s[:,None])*Dn[:,ans].T).mean(0)
    contr_top=((a*s[:,None])*Dn[:,top22].T).mean(0)
    tot=biasn.sum(0)
    print(f'L{l} recon err {err:.3f} total bias {np.round(tot,2)} proj {tot@bn:.2f}; content ans {contr_ans.sum():.2f} top22 {contr_top.sum():.2f}')
    idx=np.argsort(-proj)[:6]
    for n in idx:
        cv=(a[:,n]).std()/(abs(a[:,n].mean())+1e-6)
        print(f'   n{n}: biasproj {proj[n]:.3f} meanact {a[:,n].mean():.3f} cv {cv:.2f} cont_ans {contr_ans[n]:.3f} cont_top {contr_top[n]:.3f} bias {np.round(biasn[n],2)}')
    for n in range(7680): allres.append((l,n,proj[n],contr_ans[n],contr_top[n]))
allres=np.array(allres); np.save('work/neuron_stats.npy',allres)
o=np.argsort(-allres[:,2])
print('cum bias proj of top-k neurons (16-23):',[round(allres[o[:k],2].sum(),2) for k in [5,10,20,50,100,200,500]])
print('cum content_ans of those:',[round(allres[o[:k],3].sum(),2) for k in [5,10,20,50,100,200,500]])
print('total proj all',allres[:,2].sum(), 'global',b@bn)
o2=np.argsort(allres[:,4])
print('most top22-suppressing:',[(int(allres[i,0]),int(allres[i,1]),round(allres[i,4],3),round(allres[i,3],3)) for i in o2[:12]])
