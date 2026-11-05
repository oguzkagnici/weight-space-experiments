# Hypothesis: a letter-space correction on L22.H12's output (subtract kappa_s x [letter-s amount] from letter s+1, kappa_s =
# measured super-diagonal/diagonal ratio of H12's newline transfer matrix) removes the off-by-one co-boost; DLA screen on
# DEV-946 using H12's recorded output (o_proj input) at the answer position.
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(gold); mask = np.arange(10)[None] < nopt[:, None]
oin = np.load(D + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)
h24 = np.load(D + 'h24.npy').astype(np.float32); s_fin = np.sqrt((h24 ** 2).mean(1))
WO = load_base('model.layers.22.self_attn.o_proj.weight').numpy()
gpost = load_base('model.layers.22.post_operator_norm.weight').numpy()
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
R = (EL * wn) * gpost
s_attn = np.sqrt(((oin @ WO.T) ** 2).mean(1))
h = 12
ell = (oin[:, h * 128:(h + 1) * 128] @ (R @ WO[:, h * 128:(h + 1) * 128]).T) / (s_attn * s_fin)[:, None]  # N x 10 H12 DLA
# kappa from the newline transfer matrix (10-option questions of pass 2)
Vnl = np.load('work/p2_Vnl10.npy')[:, :, h // 4]
W = R @ WO[:, h * 128:(h + 1) * 128]
T = np.einsum('lk,jsk->sl', W, Vnl) / len(Vnl); T -= T.mean(1, keepdims=True)
kap = np.zeros(9)
for s in range(9):
    base = np.delete(T[s], [s, s + 1]).mean()
    kap[s] = (T[s, s + 1] - base) / (T[s, s] - base)
print('kappa A->B .. I->J', np.round(kap, 2))
np.save('work/kappa_h12.npy', kap)
ellc = ell - np.where(mask, ell, 0).sum(1, keepdims=True) / nopt[:, None]
def stats(L):
    L = np.where(mask, L, -1e9)
    lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
    return (L.argmax(1) == gold).mean(), (L[np.arange(N), gold] - lse).mean(), L.argmax(1)
a0, l0, p0 = stats(LP)
print(f'base acc {a0:.3f} logp {l0:.3f}; H12 DLA alone acc {(np.where(mask, ell, -1e9).argmax(1)==gold).mean():.3f}')
for c in [0.5, 1.0, 1.5]:
    dl = np.zeros((N, 10))
    for s in range(9): dl[:, s + 1] -= c * kap[s] * ellc[:, s]
    a1, l1, p1 = stats(LP + dl)
    e1 = np.where(mask, ell + dl, -1e9).argmax(1)
    print(f'c={c}: acc {a1:.3f} (w->r {((p1==gold)&(p0!=gold)).sum()}, r->w {((p1!=gold)&(p0==gold)).sum()}) logp {l1:.3f}; corrected H12 DLA alone acc {(e1==gold).mean():.3f}')
