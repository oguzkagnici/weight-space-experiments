# Hypothesis: the existing rise of re-check openers after a corrupted step (+.10) is written by a few late MLP neurons (direct logit attribution of paired activation differences).
import json, sys, numpy as np, torch
from wsp.editor import *
D='work/fwd_01'; meta=json.load(open('work/p1_meta.json'))
fam=np.array([m['fam'] for m in meta]); kind=np.array([m['kind'] for m in meta])
grp=lambda f,k:np.where((fam==f)&(kind==k))[0]
a=np.r_[grp('F1','clean'),grp('F1','clean'),grp('F2','clean')]; b=np.r_[grp('F1','big'),grp('F1','small'),grp('F2','big')]
E=load('model.embed_tokens.weight'); gF=load('model.embedding_norm.weight')
tid=lambda w:[tok.encode(x,add_special_tokens=False)[0] for x in w]
re_ids=tid(['Wait','Actually','Hmm','Hold','But','However']); go_ids=tid(['So','Thus','Then','Now','Therefore','Hence','Next'])
print('recheck ids',re_ids,'go ids',go_ids)
r=(E[re_ids].mean(0)-E[go_ids].mean(0))*gF          # logit-difference read-out direction in final-norm input space (up to the rms scale)
h24=np.load(D+'/h24.npy'); rms=np.sqrt((h24**2).mean(1)); 
ld=(h24@r.numpy())/rms; print('lens logit diff recheck-go: clean %.2f corrupt %.2f (paired diff %.2f, SE %.2f)'%(ld[a].mean(),ld[b].mean(),(ld[b]-ld[a]).mean(),(ld[b]-ld[a]).std()/np.sqrt(len(a))))
# decompose the paired difference of h24 - h20 etc.
for L in (14,18,20,21,22,23,24):
    h=np.load(D+'/h%d.npy'%L); print('  h%d paired diff along r / final rms: %.2f'%(L,((h[b]-h[a])@r.numpy()/rms[a]).mean()))
for L in (21,22,23):
    H=np.load(D+'/mlp%d.npy'%L).astype(np.float32); w2=load('model.layers.%d.feed_forward.w2.weight'%L); gp=load('model.layers.%d.post_ffn_norm.weight'%L)
    out=torch.tensor(H)@w2.T; orms=out.pow(2).mean(1).sqrt().numpy()        # pre-norm MLP output rms per text
    wr=((w2*gp[:,None]).T@r).numpy()                                         # per-neuron read-out through post_ffn_norm gain
    ca=H[a]*wr/orms[a,None]/rms[a,None]; cb=H[b]*wr/orms[b,None]/rms[b,None]; d=(cb-ca)
    tot=d.sum(1).mean(); m=d.mean(0); se=d.std(0)/np.sqrt(len(a)); top=np.argsort(-np.abs(m))[:10]
    print('L%d MLP: total paired logit-diff contribution %+.2f; top neurons:'%(L,tot))
    for j in top: print('   n%-5d %+.3f (t %.1f) | act clean %.2f corrupt %.2f | readout %+.2f | mean contribution clean %+.2f'%(j,m[j],m[j]/(se[j]+1e-9),H[a][:,j].mean(),H[b][:,j].mean(),wr[j],ca[:,j].mean()))
from sklearn.metrics import roc_auc_score
tag=np.array(['F1big']*80+['F1small']*80+['F2']*33)
for t in ('F1big','F1small','F2'):
    m=tag==t; print(t,'AUC of lens logit diff (recheck-go) as detector: %.3f'%roc_auc_score(np.r_[np.zeros(m.sum()),np.ones(m.sum())],np.r_[ld[a][m],ld[b][m]]))
