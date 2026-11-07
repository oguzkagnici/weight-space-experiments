# Hypothesis: v2a failed because after the pushed '"The' the model often continued with its own explanation ('"The interest
# created ...', '"The" answer ...') and then ended without an answer line (7 finished answers had no answer). A third quiet L23
# neuron gated on the current token being the answer-initial '"The' (the token never occurs inside the thinking, where the
# quote is ' "The') and writing ' answer' completes the line '"The answer is (X)".' that the think already ends with.
import json, numpy as np, torch
from wsp.editor import load, save_edit, tok
X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); off1=np.load('work/fwd_01/offsets.npy')
cp1=json.load(open('work/p1_closepos.json')); n1=json.load(open('work/fwd_01/meta.json'))['n_tokens']
R2=np.load('work/p2_L23_content_rows.npy').astype(np.float32); rows2=json.load(open('work/p2_L23_content_rows.json'))
meta2=json.load(open('work/p2_meta.json')); cp2={int(k):v for k,v in json.load(open('work/p2_closepos.json')).items()}; off2=np.load('work/fwd_02/offsets.npy')
o2p1=set(off2[i]+cp2[i]+1 for i,m in enumerate(meta2) if m['open']=='O2')
T=R2[[r for r in range(len(rows2)) if rows2[r] in o2p1]]
others=R2[[r for r in range(len(rows2)) if rows2[r] not in o2p1]]     # other early answer-text positions (all openers, p0..p8)
K=np.concatenate([np.arange(off1[j],off1[j]+cp1[j]) for j in range(13)])
Xk=np.asarray(X1[K],dtype=np.float32); mu=Xk.mean(0); ev,U=np.linalg.eigh(np.cov((Xk-mu).T)); P=U[:,::-1][:,:200]
cl=lambda v:v-P@(P.T@v)
b=cl(mu); b/=np.linalg.norm(b); mb=float((Xk@b).mean())
natc=np.asarray(X1[[off1[j]+cp1[j]+k for j in range(13) for k in range(1,80) if cp1[j]+k<n1[j]]],dtype=np.float32)
cfg=json.load(open('work/edit_cfg3.json')); spec=cfg['p1']
d=cl(T.mean(0)-mu); s=Xk@d; sd=s.std(); t=s.mean()+spec['zstar']*sd
w=(d-(t/mb)*b)/sd
G=lambda A:A@w
print(f"p1 gate: thinking max {G(Xk).max():.1f} (p99.99 {np.percentile(G(Xk),99.99):.1f}) | natural answer text max {G(natc).max():.1f} | other early answer positions max {G(others).max():.1f} | targets min {G(T).min():.1f} mean {G(T).mean():.1f}")
a=spec['alpha']; print(f"   alpha {a}: g thinking max {a*G(Xk).max():.1f}; target min {a*G(T).min():.1f}; n thinking tokens with g>-5: {(a*G(Xk)>-5).sum()} of {len(Xk)}")
gf=load('model.embedding_norm.weight'); gp=load('model.layers.23.post_ffn_norm.weight'); E=load('model.embed_tokens.weight')
W1=load('model.layers.23.feed_forward.w1.weight'); W3=load('model.layers.23.feed_forward.w3.weight'); W2=load('model.layers.23.feed_forward.w2.weight')
nidx=spec['neuron']
e=E[spec['tgt']]-sum(c*E[i] for i,c in spec['comp'])/sum(c for i,c in spec['comp'])
v=gf*gp*e; v=v/v.norm()
Wb=W2.bfloat16().float(); col=v*spec['col_scale']; blk=nidx//32
B=Wb[:,blk*32:(blk+1)*32].abs().clone(); oldabs=B[:,nidx%32].clone(); B[:,nidx%32]=0; mo=B.max(1).values; cap=0.98*mo
new=torch.maximum(torch.minimum(col,cap),-cap); strict=oldabs>mo; new[strict]=torch.sign(col[strict])*oldabs[strict]
print(f'   neuron {nidx}: col norm {new.norm():.3f}, cos {float(new@v/new.norm()):.3f}, rows keeping block max {int(strict.sum())}')
W1[nidx]=a*torch.tensor(w,dtype=torch.float32); W3[nidx]=torch.tensor(b,dtype=torch.float32)*spec['u']/mb; W2[:,nidx]=new
for nm,Wt in (('w1',W1),('w3',W3),('w2',W2)): print(save_edit(f'model.layers.23.feed_forward.{nm}.weight',Wt))
