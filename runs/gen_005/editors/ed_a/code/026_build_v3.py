# Hypothesis: v3 = v2 gate made sharper (alpha 16, negligible SiLU tail) and an output column clipped under each w2 row's Q8_0 peer-block absmax (no llama.cpp block-scale change), with the gain moved into the activation; it should change generation only at gated churn breaks, removing pass 3's butterfly divergences.
import json, sys, numpy as np, torch
from wsp.editor import *
PCT=float(sys.argv[1]); EFF=float(sys.argv[2]); SAVE=len(sys.argv)>3 and sys.argv[3]=='save'
ALPHA=16.0; K=2.0; L=23; N=74
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json')); rowsI=np.array([r['row'] for r in recs])
cls=np.array([r['cls'] for r in recs]); cut=np.array([r['cut'] for r in recs]); ti=np.array([r['ti'] for r in recs])
X=np.load(f'{D}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
Xb=torch.tensor(np.asarray(X[rowsI],np.float32))
isbrk=np.zeros(X.shape[0],bool); isbrk[rowsI]=True
rng=np.random.default_rng(3); samp=np.sort(rng.choice(np.where(~isbrk)[0],60000,replace=False))
Xs=torch.tensor(np.asarray(X[samp],np.float32))
mu=Xs.mean(0); mh=mu/mu.norm(); mp=float((Xs@mh).mean())
ch=Xb[cls=='churn'].mean(0); g1=ch-Xb[cls=='other'].mean(0); g2=ch-Xs.mean(0)
d=g1/g1.norm()+g2/g2.norm(); d=d/d.norm()
theta=float(np.percentile((Xb@d)[cls=='churn'].numpy(),PCT))
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight'); gp=load(f'model.layers.{L}.post_ffn_norm.weight')
ids=lambda Ls:[tok(s,add_special_tokens=False).input_ids[0] for s in Ls]
t=gf*(E[ids(['Hence','Therefore'])].mean(0)-E[ids(['Alternatively','Maybe','Perhaps','Another','Possibly'])].mean(0))
u=gp*t; u=u/u.norm()
W2o=load_original(f'model.layers.{L}.feed_forward.w2.weight'); am=torch.cat([W2o[:,64:74],W2o[:,75:96]],1).abs().max(1).values
c=torch.maximum(torch.minimum(K*u,0.98*am),-0.98*am)
S=EFF/float(c@u)          # activation gain so that act*c has the same u-component as v2's act*EFF*u
w1=ALPHA*(d-(theta/mp)*mh); w3=S*mh/(mp*ALPHA)
a=lambda Xm: (torch.nn.functional.silu(Xm@w1)*(Xm@w3)).numpy()
ab=a(Xb); as_=a(Xs); thr=0.1*S
print(f'theta {theta:.2f} S {S:.2f} ||c|| {c.norm():.3f} c.u {float(c@u):.3f}')
for cc in (0,1):
    for k in ('churn','other','commit'):
        m=(cls==k)&(cut==cc); print(f'  cut={cc} {k:6s} n {m.sum():4d} fire {(ab[m]>thr).mean():.3f} median act if fire {np.median(ab[m&(ab>thr)]) if (m&(ab>thr)).any() else 0:.2f}')
print(f'  non-break: fire {(as_>thr).mean()*100:.3f}%  min act {as_.min():.2e} (x||c|| = {as_.min()*float(c.norm()):.2e})')
if SAVE:
    W1=load_original(f'model.layers.{L}.feed_forward.w1.weight').clone(); W3=load_original(f'model.layers.{L}.feed_forward.w3.weight').clone(); W2=W2o.clone()
    W1[N]=w1; W3[N]=w3; W2[:,N]=c
    for nm,T in (('w1',W1),('w3',W3),('w2',W2)): print(save_edit(f'model.layers.{L}.feed_forward.{nm}.weight',T))
    json.dump({'theta':theta,'mp':mp,'S':S,'alpha':ALPHA,'K':K,'pct':PCT,'eff':EFF},open('work/026_params.json','w'))
