# Hypothesis: two repurposed quiet L23 neurons, gated on (i) the </think> position (p0) and (ii) the position right after
# "The answer is (X)." at the start of the visible answer (p6), can push "The" as the first answer token and <|im_end|> after
# the answer line, while staying far below threshold on every prompt/thinking token (L23 is position-local).
import json, numpy as np, torch
from wsp.editor import load, load_original, save_edit, tok
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json')); n=json.load(open(D+'meta.json'))['n_tokens']
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
think=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(13)])
Xt=np.asarray(X[think],dtype=np.float32); mK=Xt.mean(0); mKh=mK/np.linalg.norm(mK); pK=Xt@mKh
natc=np.asarray(X[[off[j]+cpos[j]+k for j in range(13) for k in range(1,40) if cpos[j]+k<n[j]]],dtype=np.float32)
def T(k): return np.asarray(X[[off[j]+cpos[j]+k for j in range(13,26)]],dtype=np.float32)
gf=load('model.embedding_norm.weight'); gp=load('model.layers.23.post_ffn_norm.weight'); E=load('model.embed_tokens.weight')
W1=load_original('model.layers.23.feed_forward.w1.weight'); W3=load_original('model.layers.23.feed_forward.w3.weight'); W2=load_original('model.layers.23.feed_forward.w2.weight')
def gate(k, natural_check=True):
    Tk=T(k); d=Tk.mean(0)-mK; best=None
    for c in np.linspace(-4,12,321):
        gK=Xt@d-c*pK; gT=Tk@d-c*(Tk@mKh)
        sc=min(-gK.max(), gT.min())/gK.std()
        if best is None or sc>best[0]: best=(sc,c,gK.max(),gT.min(),gK.std(),gT.mean())
    sc,c,kmax,tmin,ksd,tmean=best
    w=d-c*mKh
    gN=natc@w
    print(f'p{k}: c {c:.2f} thinking max {kmax:.1f} sd {ksd:.1f} | target min {tmin:.1f} mean {tmean:.1f} | natural content max {gN.max():.1f} (n>0: {(gN>0).sum()} of {len(gN)}) | margin {sc:.1f} sd')
    return w, kmax, tmin, tmean
def write_dir(tgt, comp):
    # residual (pre post_ffn_norm) direction that raises logit(tgt) against prob-weighted competitors
    e=E[tgt]-sum(w*E[c] for c,w in comp)/sum(w for c,w in comp)
    return (gf*gp*e)
def q8_col(W2, nidx, v, scale):
    W=W2.bfloat16().float(); col=v*scale
    blk=nidx//32; B=W[:,blk*32:(blk+1)*32].abs().clone(); oldabs=B[:,nidx%32].clone(); B[:,nidx%32]=0
    mo=B.max(1).values; cap=0.98*mo
    new=torch.maximum(torch.minimum(col,cap),-cap)
    strict=oldabs>mo        # old value was the block max: keep the same magnitude so the block scale is unchanged
    new[strict]=torch.sign(col[strict])*oldabs[strict]
    new[strict & (col==0)]=W[strict & (col==0),nidx]
    print(f'   neuron {nidx}: rows clipped {(col.abs()>cap).sum().item()}, rows with old block max {strict.sum().item()}, col norm {new.norm():.3f}, cos(new,v) {float(new@v/new.norm()/v.norm()):.3f}')
    return new
cfg=json.load(open('work/edit_cfg.json'))
for spec in cfg['neurons']:
    k=spec['pos']; nidx=spec['neuron']
    w,kmax,tmin,tmean=gate(k)
    w=torch.tensor(w,dtype=torch.float32)
    alpha=spec['g_target']/tmean          # gate pre-activation at the mean target
    print(f'   alpha {alpha:.4f}: g at thinking max {alpha*kmax:.1f}, at target min {alpha*tmin:.1f}')
    dn=w/ w.norm()
    uT=float((torch.tensor(T(k))@dn).mean())
    w3=dn*spec['u_target']/uT            # up value u_target at the mean target
    v=write_dir(spec['tgt'], spec['comp'])
    v=v/v.norm()
    col=q8_col(W2, nidx, v, spec['col_scale'])
    W1[nidx]=alpha*w; W3[nidx]=w3; W2[:,nidx]=col
print(save_edit('model.layers.23.feed_forward.w1.weight',W1))
print(save_edit('model.layers.23.feed_forward.w3.weight',W3))
print(save_edit('model.layers.23.feed_forward.w2.weight',W2))
