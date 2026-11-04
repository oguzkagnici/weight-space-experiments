# Hypothesis: the model has a content-independent early-letter bias at the answer position; find which layer/component writes it (logit lens + linear decomposition of final letter logits).
from wsp.editor import *
import numpy as np, json, torch
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions()
nopt=np.array([len(Q[r['id']]['options']) for r in rows]); ans=np.array([r['answer'] for r in rows])
P=np.array([r['probs']+[0.0]*(10-len(r['probs'])) for r in rows]); corr=np.array([r['correct'] for r in rows])
m10=nopt==10
print('n10',m10.sum(),'acc10',corr[m10].mean())
# mean log prob per letter (10-opt)
lp=np.log(P[m10]+1e-9); lpc=lp-lp.mean(1,keepdims=True)
print('mean centered logp per letter', np.round(lpc.mean(0),2))
# near-miss stats
rank=(P>P[np.arange(len(P)),ans][:,None]).sum(1)
print('rank of correct among wrong:', np.bincount(rank[corr==0],minlength=10))
print('pmax on wrong: median',np.median(P[corr==0].max(1)))
# position of predicted vs answer on wrong
E=load_original('model.embed_tokens.weight')[LETTER_IDS].numpy()  # 10x2048
wn=load_original('model.embedding_norm.weight').numpy()
def lens(h):
    h=h.astype(np.float32); r=np.sqrt((h**2).mean(1,keepdims=True)+1e-5)
    return (h/r*wn)@E.T
for l in range(0,25,2) if False else [8,10,12,14,16,17,18,19,20,21,22,23,24]:
    h=np.load(D+f'h{l}.npy'); L=lens(h)
    L=np.where(np.arange(10)[None]<nopt[:,None],L,-1e9)
    acc=(L.argmax(1)==ans).mean()
    Lc=L[m10]-L[m10].mean(1,keepdims=True)
    print(f'lens h{l}: acc {acc:.3f} bias {np.round(Lc.mean(0),1)}')
# decomposition at final: logit_k = sum_c c.(wn*E_k)/rms(h24)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
U=(wn*E).T  # 2048x10
Uc=U-U.mean(1,keepdims=True)
comps={'emb':np.load(D+'h0.npy')}
for l in range(24):
    comps[f'L{l}.mix']=np.load(D+f'out_model.layers.{l}.post_operator_norm.npy')
    comps[f'L{l}.mlp']=np.load(D+f'out_model.layers.{l}.post_ffn_norm.npy')
tot=sum(c.astype(np.float32) for c in comps.values())
print('recon err',np.abs(tot-h24).max(), np.abs(h24).max())
res=[]
for k,c in comps.items():
    contrib=(c.astype(np.float32)@Uc)/r   # n x10 centered
    bias=contrib[m10].mean(0)
    # content: mean contribution to correct letter minus mean over letters
    cont=contrib[np.arange(len(ans)),ans].mean()
    res.append((k,bias,cont, np.abs(bias).max()))
res.sort(key=lambda x:-x[3])
for k,b,c,_ in res[:16]:
    print(f'{k:10s} content {c:6.2f} bias {np.round(b,2)}')
np.save('work/lens_U.npy',U)
