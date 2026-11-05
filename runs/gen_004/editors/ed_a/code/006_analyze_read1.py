# Hypothesis: at post-conclusion breaks the next-token mass on reconsider-openers (But/Let/Check/However/Option/Now/Wait...) vs commit-openers (Hence/Therefore/Thus/So/Final) is written by a few late MLP neurons / heads; measure P(R), P(C) and locate the writers by direct logit attribution.
import json, numpy as np, torch
from wsp.editor import *
K=1; out='work/fwd_01'
meta=json.load(open(f'{out}/meta.json')); items=json.load(open('work/read1_items.json'))
R=[3911,7731,12108,8952,23671,8175,37967,93222,91780,39188,29747]
C=[43129,25705,21009,4597,24081]
ids=np.load(f'{out}/topk_ids.npy'); lp=np.load(f'{out}/topk_logprobs.npy')
def mass(row_ids,row_lp,S):
    return float(sum(np.exp(l) for i,l in zip(row_ids,row_lp) if i in S))
pR=np.array([mass(ids[j],lp[j],set(R)) for j in range(len(items))]); pC=np.array([mass(ids[j],lp[j],set(C)) for j in range(len(items))])
grp=lambda it:(it['kind'],it['label'] if it['kind']=='concl' else 'ctrl',it['cut'])
import collections
G=collections.defaultdict(list)
for j,it in enumerate(items): G[grp(it)].append(j)
for g,js in sorted(G.items()):
    print(g,len(js),'P(R) %.3f  P(C) %.3f'%(pR[js].mean(),pC[js].mean()))
# top tokens overall at conclusion points
cnt=collections.Counter()
for j,it in enumerate(items):
    if it['kind']=='concl':
        for i,l in zip(ids[j][:10],lp[j][:10]): cnt[int(i)]+=float(np.exp(l))
print('top next tokens at conclusion breaks:',[(tok.decode([i]).strip(),round(v/80,3)) for i,v in cnt.most_common(15)])
np.save('work/r1_pR.npy',pR); np.save('work/r1_pC.npy',pC)
# logit lens on direction d = mean_R emb - mean_C emb (through final norm)
E_full=load('model.embed_tokens.weight'); E=E_full; gf=load('model.embedding_norm.weight')
dvec=(E[R].mean(0)-E[C].mean(0))
H={l:torch.tensor(np.load(f'{out}/h{l}.npy').astype(np.float32)) for l in range(25)}
def lens(h): return ((h/ h.pow(2).mean(-1,keepdim=True).add(1e-5).sqrt())*gf) @ dvec
conc=[j for j,it in enumerate(items) if it['kind']=='concl']; ctrl=[j for j,it in enumerate(items) if it['kind']=='ctrl']
print('logit-lens R-C by layer (concl mean / ctrl mean):')
print(' '.join('%d:%.1f/%.1f'%(l,lens(H[l][conc]).mean(),lens(H[l][ctrl]).mean()) for l in range(8,25)))
# check the lens reproduces actual logits at final layer for one token
lf=((H[24]/H[24].pow(2).mean(-1,keepdim=True).add(1e-5).sqrt())*gf)@E[ids[:,0]].T
print('check final: top1 logprob vs lens diag spread', float(np.diag(lf.numpy()).mean()))
# DLA per block: contribution of each layer's MLP and mixer to the R-C lens (final rms fixed)
rf=H[24].pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
u=(gf*dvec)  # direction in residual
res=[]
for l in range(24):
    W2=load(f'model.layers.{l}.feed_forward.w2.weight'); gp=load(f'model.layers.{l}.post_ffn_norm.weight')
    a=torch.tensor(np.load(f'{out}/mlp{l}.npy').astype(np.float32))
    o=a@W2.T; ro=o.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt()
    y=gp*o/ro
    mlp_c=(y@u)/rf[:,0]
    tot=((H[l+1]-H[l])@u)/rf[:,0]
    mix_c=tot-mlp_c
    res.append((l,mlp_c,mix_c))
    # per-neuron contributions (linear given ro)
    nc=(a*((W2.T*gp)@u)[None,:])/ro/rf  # [n,7680]
    np.save(f'work/r1_neuron_contrib_L{l}.npy',nc.numpy().astype(np.float32))
print('per-block contribution to R-C (concl mean, ctrl mean, concl reconsider-commit):')
rec=[j for j in conc if items[j]['label']=='reconsider']; com=[j for j in conc if items[j]['label']=='commit']
for l,m,x in res:
    print('L%2d mlp %+.2f %+.2f %+.2f | mixer %+.2f %+.2f %+.2f'%(l,m[conc].mean(),m[ctrl].mean(),m[rec].mean()-m[com].mean(),x[conc].mean(),x[ctrl].mean(),x[rec].mean()-x[com].mean()))
