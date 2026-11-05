# Hypothesis: the L22/L23 per-letter "leader suppressor" neurons (gen 1) suppress late letters harder than A/B; scaling
# their direct letter readout down / equalising it should help on MY DEV set too (DLA screen, direct effect only).
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(gold); mask = np.arange(10)[None] < nopt[:, None]
h24 = np.load(D + 'h24.npy').astype(np.float32); s_fin = np.sqrt((h24 ** 2).mean(1))
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
def stats(L):
    L = np.where(mask, L, -1e9)
    lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
    return (L.argmax(1) == gold).mean(), (L[np.arange(N), gold] - lse).mean(), L.argmax(1)
a0, l0, p0 = stats(LP)
print(f'base acc {a0:.3f} logp {l0:.3f}')
contrib = {}
for layer in [22, 23]:
    act = np.load(D + f'mlp{layer}.npy').astype(np.float32)
    W2 = load_base(f'model.layers.{layer}.feed_forward.w2.weight').numpy()
    g = load_base(f'model.layers.{layer}.post_ffn_norm.weight').numpy()
    tot = act @ W2.T; s_m = np.sqrt((tot ** 2).mean(1))
    Rd = (EL * wn * g) @ W2  # 10 x 7680 letter readout per neuron
    contrib[layer] = (act[:, None, :] * Rd[None]) / (s_m * s_fin)[:, None, None]  # N x 10 x 7680
sup22 = [1812, 464, 7357, 6492, 3344, 397, 2786, 3310]  # C..J
sup22ab = [389, 6620, 5859]  # A, B (gen1 ed_a)
boost = [(22, 2212), (23, 1685), (23, 2595), (23, 1439)]
for nm, sel, alpha in [('L22 C-J suppressors x0.5', [(22, n) for n in sup22], 0.5),
                       ('L22 C-J suppressors x0', [(22, n) for n in sup22], 1.0),
                       ('L22 C-J supp x0.5 + boosters x0.5', [(22, n) for n in sup22] + boost, 0.5),
                       ('L22 A-J suppressors x0.5', [(22, n) for n in sup22 + sup22ab], 0.5)]:
    dl = np.zeros((N, 10))
    for l, n in sel: dl -= alpha * contrib[l][:, :, n]
    a1, l1, p1 = stats(LP + dl)
    print(f'{nm:38s}: acc {a1:.3f} (w->r {((p1==gold)&(p0!=gold)).sum()}, r->w {((p1!=gold)&(p0==gold)).sum()}) logp {l1:.3f}')
