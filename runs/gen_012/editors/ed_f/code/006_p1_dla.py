# Hypothesis H1: at forced-answer probes in thinking context, some late-layer write (block or part of a w2 spectrum) lowers the gold letter; locate it by direct read-out, choose on split A, confirm on split B.
import json, numpy as np, torch
from wsp.editor import *
torch.set_num_threads(3)
D='work/fwd_01'
P=json.load(open('work/probes.json'))[:160]
L='ABCDEFGHIJ'; Q=questions()
gold=np.array([L.index(p['gold']) for p in P]); nopt=np.array([len(Q[p['id']]['options']) for p in P])
split=np.array([p['split'] for p in P]); src=np.array([p['src'] for p in P])
E=load_original('model.embed_tokens.weight')[LETTER_IDS].double()   # 10 x 2048
gf=load_original('model.embedding_norm.weight').double()
H=[torch.tensor(np.load(f'{D}/h{l}.npy')).double() for l in range(25)]
def fnorm(x,eps=1e-5): return x/torch.sqrt((x*x).mean(-1,keepdim=True)+eps)*gf
def letters(h):
    z=fnorm(h)@E.T
    for i in range(len(P)): z[i,nopt[i]:]=-1e9
    return z
z=letters(H[24]); lp=torch.log_softmax(z,-1)
# check against top-k
ids=np.load(f'{D}/topk_ids.npy'); tl=np.load(f'{D}/topk_logprobs.npy')
d=[]
for i in range(len(P)):
    for j in range(2):
        if ids[i,j] in LETTER_IDS:
            k=LETTER_IDS.index(ids[i,j]); 
            for j2 in range(j+1,5):
                if ids[i,j2] in LETTER_IDS:
                    k2=LETTER_IDS.index(ids[i,j2]); d.append(((z[i,k]-z[i,k2]).item(),tl[i,j]-tl[i,j2])); break
            break
d=np.array(d); print('lens check: n',len(d),'max |logit diff err| %.3f'%np.abs(d[:,0]-d[:,1]).max(), 'letters mass (top20) mean %.3f'%np.mean([np.exp(tl[i][np.isin(ids[i],LETTER_IDS)]).sum() for i in range(len(P))]))
pred=z.argmax(-1).numpy(); pg=lp[np.arange(len(P)),gold].exp().numpy()
rank=(z>z[np.arange(len(P)),gold][:,None]).sum(-1).numpy()
for s in ('bank40','pk1_15'):
    for sp in 'AB':
        m=(src==s)&(split==sp); print(s,sp,'n',m.sum(),'acc %.3f P(gold) %.3f rank %.2f'%((pred==gold)[m].mean(),pg[m].mean(),rank[m].mean()))
fin=np.array([p['pred']==p['gold'] for p in P]); print('final trace right %.3f; probe=final pick %.3f'%(fin.mean(),np.mean([L[pred[i]]==P[i]['pred'] for i in range(len(P))])))
print('logit lens by layer: acc / P(gold):',' '.join('%d:%.2f/%.2f'%(l,(letters(H[l]).argmax(-1).numpy()==gold).mean(),torch.log_softmax(letters(H[l]),-1)[np.arange(len(P)),gold].exp().mean()) for l in (12,14,16,18,20,21,22,23,24)))
# margin read-out direction per probe (linear at fixed final-norm scale)
def margin(zz):
    out=[]
    for i in range(len(P)):
        o=[k for k in range(nopt[i]) if k!=gold[i]]; out.append(zz[i,gold[i]]-zz[i,o].mean())
    return torch.stack(out)
scale=1/torch.sqrt((H[24]**2).mean(-1,keepdim=True)+1e-5)
def contrib(x):  # x: residual write (n,2048) -> margin contribution
    zz=(x*scale*gf)@E.T; return margin(zz)
print('total margin mean %.2f'%margin(z.clamp(min=-50)).mean())
rows=[]
for l in range(12,24):
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').double(); a=torch.tensor(np.load(f'{D}/mlp{l}.npy')).double()
    pre=a@W2.T; g=load_original(f'model.layers.{l}.post_ffn_norm.weight').double()
    mo=pre/torch.sqrt((pre*pre).mean(-1,keepdim=True)+1e-5)*g
    blk=H[l+1]-H[l]; mix=blk-mo
    cm=contrib(mo).numpy(); cx=contrib(mix).numpy()
    print('L%d mlp: margin %+.2f (se %.2f) [A %+.2f B %+.2f] | mixer: %+.2f (se %.2f) | |mlp| %.1f |mix| %.1f |h| %.1f'%(l,cm.mean(),cm.std()/np.sqrt(len(P)),cm[split=='A'].mean(),cm[split=='B'].mean(),cx.mean(),cx.std()/np.sqrt(len(P)),mo.norm(dim=-1).mean(),mix.norm(dim=-1).mean(),H[l+1].norm(dim=-1).mean()))
