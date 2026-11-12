# Hypothesis: the early-letter ramp at decision positions is written by a small set of near-constant MLP neurons; rank neurons by mean DLA ramp slope at the 82 probe positions and check their activation consistency.
import json, numpy as np
from wsp.editor import *
d='work/fwd_01'
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy(); g=load('model.embedding_norm.weight').numpy()
h24=np.load(f'{d}/h24.npy'); rms24=np.sqrt((h24**2).mean(1)+1e-5)
r=(np.arange(10)-4.5); r=r/ (r@r)   # slope estimator: slope = r . v
res=[]
for l in range(14,24):
    w2=load(f'model.layers.{l}.feed_forward.w2.weight').numpy(); pg=load(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    A=np.load(f'{d}/mlp{l}.npy')            # 82 x 7680
    out=A@w2.T; rmso=np.sqrt((out**2).mean(1)+1e-5)  # 82
    Vcol=(E*g)@(pg[:,None]*w2)               # 10 x 7680
    slope_col=r@ (Vcol-Vcol.mean(0))          # per unit activation
    C=A*slope_col[None,:]/rmso[:,None]/rms24[:,None]   # 82 x 7680 slope contributions
    m=C.mean(0); s=C.std(0)
    print(l,'layer total slope %.3f'%C.sum(1).mean())
    for j in np.argsort(m)[:6]: res.append((l,int(j),float(m[j]),float(s[j]),float(A[:,j].mean()),float(A[:,j].std())))
res.sort(key=lambda x:x[2])
for x in res[:25]: print('L%d n%d slope %.4f sd %.4f act mean %.3f sd %.3f'%x)
json.dump(res,open('work/ramp_neurons.json','w'))
