# Hypothesis: the leader-suppressor bank of L22 is asymmetric (A/B neurons weak). Equalising UPWARD (scaling the direct
# letter readout of the A/B suppressors up to the C-J strength) removes the early-letter advantage without the calibration
# cost of removing suppression. DLA screen on DEV-946 (direct effect only).
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(gold); mask = np.arange(10)[None] < nopt[:, None]
h24 = np.load(D + 'h24.npy').astype(np.float32); s_fin = np.sqrt((h24 ** 2).mean(1))
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
act = np.load(D + 'mlp22.npy').astype(np.float32)
W2 = load_base('model.layers.22.feed_forward.w2.weight').numpy()
g = load_base('model.layers.22.post_ffn_norm.weight').numpy()
tot = act @ W2.T; s_m = np.sqrt((tot ** 2).mean(1))
Rd = (EL * wn * g) @ W2
Rdc = Rd - Rd.mean(0)
bank = {0: [389], 1: [6620, 5859], 2: [1812], 3: [464], 4: [7357], 5: [6492], 6: [3344], 7: [397], 8: [2786], 9: [3310]}
L0 = np.where(mask, LP, -1e9); lead = L0.argmax(1)
print('letter | neurons | own-letter readout (centered) | mean act when letter leads | mean own push when leads')
push = {}
for s, ns in bank.items():
    m = lead == s
    pp = sum(act[m, n] * Rdc[s, n] / (s_m[m] * s_fin[m]) for n in ns)
    push[s] = pp.mean()
    print(f'  {"ABCDEFGHIJ"[s]} | {ns} | {[round(float(Rdc[s, n]), 2) for n in ns]} | {[round(float(act[m, n].mean()), 2) for n in ns]} | {pp.mean():+.2f} (n={m.sum()})')
def stats(L):
    L = np.where(mask, L, -1e9)
    lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
    return (L.argmax(1) == gold).mean(), (L[np.arange(N), gold] - lse).mean(), L.argmax(1)
a0, l0, p0 = stats(LP)
contrib = (act[:, None, :] * Rd[None]) / (s_m * s_fin)[:, None, None]
target = np.mean([push[s] for s in range(2, 10)])
print(f'base acc {a0:.3f} logp {l0:.3f}; mean C-J push {target:+.2f}')
for kA, kB in [(1.5, 1.5), (2, 2), (3, 3), (target / push[0], target / push[1])]:
    dl = (kA - 1) * contrib[:, :, 389] + (kB - 1) * (contrib[:, :, 6620] + contrib[:, :, 5859])
    a1, l1, p1 = stats(LP + dl)
    print(f'A x{kA:.2f}, B x{kB:.2f}: acc {a1:.3f} (w->r {((p1==gold)&(p0!=gold)).sum()}, r->w {((p1!=gold)&(p0==gold)).sum()}) logp {l1:.3f} counts {np.bincount(p1, minlength=10)}')
