# Hypothesis: removing the direct letter-logit contribution of the top correct-suppressing MLP22 neurons (linear DLA estimate, ignores layer-23 downstream effects) raises accuracy; screen k and strength before spending passes.
import json, numpy as np
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows]); cor = np.array([r['correct'] for r in rows])
mask = np.arange(10)[None] < nopt[:, None]
E = np.load('work/E_letters.npy'); w = load('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms24 = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T
L = 22
act = np.load(A + f'mlp{L}.npy').astype(np.float32)
W2 = load(f'model.layers.{L}.feed_forward.w2.weight').numpy(); pw = load(f'model.layers.{L}.post_ffn_norm.weight').numpy()
y = act @ W2.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
G = (pw[:, None] * W2).T @ U
help_n = np.load('work/mlp22_help.npy'); o = np.argsort(help_n)
LL = np.load('work/lens_p1.npy')[24]
def ev(lg):
    lg = np.where(mask, lg, -1e9)
    p = lg.argmax(1); acc = (p == ans).mean()
    lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
    return acc, (lg[np.arange(n), ans] - lse).mean(), np.bincount(p, minlength=10)
print('base', ev(LL))
for name, idx in [('top8', o[:8]), ('top5', o[:5]), ('top10', o[:10]), ('top20', o[:20]), ('top3', o[:3])]:
    c = (act[:, idx] @ G[idx]) / (rq * rms24)[:, None]
    for a in [0.5, 1.0, 1.5]:
        r = ev(LL - a * c)
        print(name, a, round(r[0], 4), round(r[1], 3), r[2])
# per-neuron single removal
print('single-neuron removal acc delta (x1):')
for j in o[:10]:
    c = np.outer(act[:, j], G[j]) / (rq * rms24)[:, None]
    r = ev(LL - c); print(j, round(r[0] - cor.mean(), 4), round(r[1] - ev(LL)[1], 3))
print('letter profile (centered G) of top suppressors:')
for j in o[:8]:
    g = G[j] - G[j].mean(); print(j, np.round(g, 3), 'act sign frac>0', (act[:, j] > 0).mean().round(2))
