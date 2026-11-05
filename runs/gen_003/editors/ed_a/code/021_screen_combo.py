# Hypothesis: the letter-stage loss is mostly the asymmetric L22 leader-suppressor bank; full removal (or equalisation to
# the A/B level) of the C-J suppressors' letter readout recovers ~+2 points; check dose, split-half stability, and adding
# the near-constant L22/L23 prior writers (gen 1) on top. DLA screen on DEV-946 (direct effect only).
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); Q = questions()
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(gold); mask = np.arange(10)[None] < nopt[:, None]
h24 = np.load(D + 'h24.npy').astype(np.float32); s_fin = np.sqrt((h24 ** 2).mean(1))
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
contrib = {}
for layer in [22, 23]:
    act = np.load(D + f'mlp{layer}.npy').astype(np.float32)
    W2 = load_base(f'model.layers.{layer}.feed_forward.w2.weight').numpy()
    g = load_base(f'model.layers.{layer}.post_ffn_norm.weight').numpy()
    tot = act @ W2.T; s_m = np.sqrt((tot ** 2).mean(1))
    Rd = (EL * wn * g) @ W2
    contrib[layer] = (act[:, None, :] * Rd[None]) / (s_m * s_fin)[:, None, None]
def stats(L, sub=None):
    sub = np.ones(N, bool) if sub is None else sub
    L = np.where(mask, L, -1e9)[sub]; gg = gold[sub]
    lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
    return (L.argmax(1) == gg).mean(), (L[np.arange(len(gg)), gg] - lse).mean(), L.argmax(1)
sup = [1812, 464, 7357, 6492, 3344, 397, 2786, 3310]
prior22 = [190, 1449, 2820, 4543, 5653, 2212, 3998]
prior23 = [1047, 4597, 1903, 6518, 4531, 514, 3073, 2189, 1439, 7466]
half = np.arange(N) % 2 == 0
a0, l0, p0 = stats(LP)
print(f'base acc {a0:.3f} logp {l0:.3f}')
for nm, f, extra in [('C-J f=0.5', .5, []), ('C-J f=0.25', .25, []), ('C-J f=0', 0, []), ('C-J f=-0.25', -.25, []),
                     ('C-J f=0 + A/B supp f=0', 0, [(22, 389), (22, 6620), (22, 5859)]),
                     ('C-J f=0 + prior L22/L23 removed', 0, [(22, n) for n in prior22] + [(23, n) for n in prior23]),
                     ('prior L22/L23 removed only', 1, [(22, n) for n in prior22] + [(23, n) for n in prior23])]:
    dl = np.zeros((N, 10))
    for n in sup: dl -= (1 - f) * contrib[22][:, :, n]
    for l, n in extra: dl -= contrib[l][:, :, n]
    a1, l1, p1 = stats(LP + dl)
    h1 = stats(LP + dl, half)[0] - stats(LP, half)[0]; h2 = stats(LP + dl, ~half)[0] - stats(LP, ~half)[0]
    p0n = np.where(mask, LP, -1e9).argmax(1)
    cats = np.array([Q[q]['category'] for q in ids])
    print(f'{nm:34s}: acc {a1:.3f} ({a1-a0:+.3f}; halves {h1:+.3f} {h2:+.3f}) w->r {((p1==gold)&(p0n!=gold)).sum()} r->w {((p1!=gold)&(p0n==gold)).sum()} logp {l1:.3f} counts {np.bincount(p1, minlength=10)}')
