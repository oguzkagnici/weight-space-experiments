# Hypothesis: with a bias-free gate g = d.x - c*(mK.x) (d = mean-diff, mK = thinking-mean direction), the target positions
# {p0,p1} and {p6} are separated from all prompt+thinking tokens with a usable margin, and natural-content positions do not
# trigger the p6 (EOS) gate.
import json, numpy as np
from wsp.editor import tok
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json')); n=json.load(open(D+'meta.json'))['n_tokens']
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
TV=np.load(D+'token_logprobs.npy')
think=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(13)])
Xt=np.asarray(X[think],dtype=np.float32); mK=Xt.mean(0); mKh=mK/np.linalg.norm(mK)
pK=Xt@mKh
def tgt(ks): return np.asarray(X[[off[j]+cpos[j]+k for j in range(13,26) for k in ks]],dtype=np.float32)
nat_content=lambda k0,k1: np.asarray(X[[off[j]+cpos[j]+k for j in range(13) for k in range(k0,k1) if cpos[j]+k<n[j]]],dtype=np.float32)
for name,ks in (('A p0p1',[0,1]),('A p0',[0]),('B p6',[6]),('B p5p6',[5,6])):
    T=tgt(ks); d=T.mean(0)-mK
    sT=T@d; pT=T@mKh; sK=Xt@d
    best=None
    for c in np.linspace(0,8,161):
        gK=sK-c*pK; gT=sT-c*pT
        m=(gT.min()-gK.max())/gK.std()
        if best is None or m>best[0]: best=(m,c,gK.max(),gT.min(),gK.std(),gK.mean())
    m,c,kmax,tmin,ksd,kmean=best
    print(f'{name}: best c {c:.2f}: thinking max {kmax:.1f} (mean {kmean:.1f}, sd {ksd:.2f}) target min {tmin:.1f} -> margin {m:.1f} sd')
    # natural content positions p1..p12 (texts 0-12)
    for k0,k1 in ((1,2),(2,7),(7,13)):
        N=nat_content(k0,k1); gN=N@d-c*(N@mKh); print(f'    natural content p{k0}-{k1-1}: max {gN.max():.1f} mean {gN.mean():.1f}')
    np.save(f'work/p1_gate_{name.split()[1]}.npy',np.concatenate([d,[c]]))
# exact baseline log-probs at variant positions (teacher-forced next token)
for j in range(13,26):
    c0=off[j]+cpos[j]-j   # token_logprobs has n-1 entries per text -> offset shift by j
    print(j, 'lp(" answer"|The) %.2f  lp(" is") %.2f  lp(letter) %.2f lp(").") %.2f'%(TV[c0+1],TV[c0+2],TV[c0+4],TV[c0+5]))
