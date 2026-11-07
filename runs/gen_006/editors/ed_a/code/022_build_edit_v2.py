# Hypothesis: two repurposed quiet L23 neurons (position-local, last layer) can (i) at the </think> position push the single
# token '"The' (the think's own closing line is '..."The answer is (X)".', so the answer line then follows by copying, P~e^-3)
# and (ii) right after '"The answer is (X)".' (6 tokens after </think>) push <|im_end|>, while staying far below threshold on
# every prompt/thinking token, including the think's own '"The answer is (X)".' lines (which must still be followed by </think>).
import json, numpy as np, torch
from wsp.editor import load, load_original, save_edit, tok
X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); off1=np.load('work/fwd_01/offsets.npy')
cp1=json.load(open('work/p1_closepos.json')); n1=json.load(open('work/fwd_01/meta.json'))['n_tokens']
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); off2=np.load('work/fwd_02/offsets.npy')
cp2={int(k):v for k,v in json.load(open('work/p2_closepos.json')).items()}; meta2=json.load(open('work/p2_meta.json'))
t1=json.load(open('work/req_p1.json'))['texts']
F=lambda X,rows: np.asarray(X[rows],dtype=np.float32)
K=np.concatenate([np.arange(off1[j],off1[j]+cp1[j]) for j in range(13)])
Xk=F(X1,K); mu=Xk.mean(0); ev,U=np.linalg.eigh(np.cov((Xk-mu).T)); P=U[:,::-1][:,:200]
cl=lambda v:v-P@(P.T@v)
b=cl(mu); b/=np.linalg.norm(b); mb=float((Xk@b).mean())
# hard negatives: thinking tokens that end a '"The answer is (X)".' line, i.e. the token right before </think> and all '".' tokens
ids1=[tok(t,add_special_tokens=False)['input_ids'] for t in t1[:13]]
hard=np.array([off1[j]+i for j in range(13) for i in range(cp1[j]) if ids1[j][i]==2784 or i==cp1[j]-1])
natc=np.array([off1[j]+cp1[j]+k for j in range(13) for k in range(1,80) if cp1[j]+k<n1[j]])
T0=F(X1,[off1[j]+cp1[j] for j in range(13,26)])
o2=[i for i,m in enumerate(meta2) if m['open']=='O2']
T6=F(X2,[off2[i]+cp2[i]+6 for i in o2])
other6=F(X2,[off2[i]+cp2[i]+k for i,m in enumerate(meta2) if m['open']!='O2' for k in range(1,9) if cp2[i]+k<off2[i+1]-off2[i]])
gf=load('model.embedding_norm.weight'); gp=load('model.layers.23.post_ffn_norm.weight'); E=load('model.embed_tokens.weight')
W1=load_original('model.layers.23.feed_forward.w1.weight'); W3=load_original('model.layers.23.feed_forward.w3.weight'); W2=load_original('model.layers.23.feed_forward.w2.weight')
def q8_col(nidx, v, scale):
    W=W2.bfloat16().float(); col=v*scale; blk=nidx//32
    B=W[:,blk*32:(blk+1)*32].abs().clone(); oldabs=B[:,nidx%32].clone(); B[:,nidx%32]=0
    mo=B.max(1).values; cap=0.98*mo
    new=torch.maximum(torch.minimum(col,cap),-cap)
    strict=oldabs>mo; new[strict]=torch.sign(col[strict])*oldabs[strict]
    print(f'   neuron {nidx}: rows clipped {(col.abs()>cap).sum().item()}, rows keeping old block max {strict.sum().item()}, col norm {new.norm():.3f}, cos(new,v) {float(new@v/new.norm()/v.norm()):.3f}')
    return new
cfg=json.load(open('work/edit_cfg2.json'))
for spec,T in zip(cfg['neurons'],(T0,T6)):
    d=cl(T.mean(0)-mu); s=Xk@d; sd=s.std(); t=s.mean()+spec['zstar']*sd
    w=(d-(t/mb)*b)/sd          # gate score in thinking-sd units, 0 at threshold
    G=lambda A:A@w
    print(f"neuron {spec['neuron']} (target p{spec['pos']}): thinking max {G(Xk).max():.1f} mean {G(Xk).mean():.1f} | hard negatives max {G(F(X1,hard)).max():.1f} | natural answer text max {G(F(X1,natc)).max():.1f} | other openers p1-8 max {G(other6).max():.1f} | targets min {G(T).min():.1f} mean {G(T).mean():.1f}")
    alpha=spec['alpha']
    print(f"   alpha {alpha}: g thinking max {alpha*G(Xk).max():.1f}, hard max {alpha*G(F(X1,hard)).max():.1f}, target min {alpha*G(T).min():.1f}")
    w=torch.tensor(w,dtype=torch.float32)
    up=torch.tensor(b,dtype=torch.float32)*spec['u']/mb      # up ~ constant u on all tokens
    e=E[spec['tgt']]-sum(c*E[i] for i,c in spec['comp'])/sum(c for i,c in spec['comp'])
    v=gf*gp*e; v=v/v.norm()
    col=q8_col(spec['neuron'], v, spec['col_scale'])
    W1[spec['neuron']]=alpha*w; W3[spec['neuron']]=up; W2[:,spec['neuron']]=col
for nm,Wt in (('w1',W1),('w3',W3),('w2',W2)): print(save_edit(f'model.layers.23.feed_forward.{nm}.weight',Wt))
