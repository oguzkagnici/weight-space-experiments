# Hypothesis: some L23 neurons have a tiny logit-level footprint in the thinking (mean |a| x logit spread of their w2 column
# through post_ffn_norm and final norm gains), so repurposing them perturbs the thinking least.
import json, numpy as np, torch
from wsp.editor import load
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json'))
A=np.load(D+'mlp23.npy',mmap_mode='r')
sa=np.zeros(7680); sa2=np.zeros(7680); mx=np.zeros(7680); T=0
for j in range(13):
    a=np.abs(np.asarray(A[off[j]:off[j]+cpos[j]],dtype=np.float32)); sa+=a.sum(0); sa2+=(a**2).sum(0); mx=np.maximum(mx,a.max(0)); T+=len(a)
ma=sa/T; ra=np.sqrt(sa2/T)
W2=load('model.layers.23.feed_forward.w2.weight'); gp=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight')
E=load('model.embed_tokens.weight')
idx=torch.randperm(E.shape[0],generator=torch.Generator().manual_seed(0))[:16000]
Es=E[idx]
V=(gf*gp)[:,None]*W2       # 2048x7680
L=Es@V                      # 16000x7680
spread=L.std(0).numpy()
loud=ra*spread
o=np.argsort(loud)
print('quietest L23 neurons: idx, rms|a|, max|a|, logit spread of column, loudness')
for i in o[:15]: print(int(i), f'{ra[i]:.4f} {mx[i]:.3f} {spread[i]:.3f} {loud[i]:.5f}')
print('median loudness %.4f; n74 %.5f'%(np.median(loud),loud[74]))
np.save('work/p1_loud23.npy',loud)
