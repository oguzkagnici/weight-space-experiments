# Hypothesis: removing the letter-subspace push of the few late-MLP neurons that write a near-constant early-letter offset (mu>thr, sd<mu, small correct-help) removes most of the early-letter bias and raises accuracy (linear DLA counterfactual; neurons chosen on half of DEV-512, tested on the other half).
import json, numpy as np
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows])
mask = np.arange(10)[None] < nopt[:, None]; m10 = nopt == 10
E = np.load('work/E_letters.npy'); w = load('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms24 = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T; Uc = U - U.mean(1, keepdims=True)
LL = np.load('work/lens_p1.npy')[24]
def ev(lg, idx):
    lg = np.where(mask, lg, -1e9)[idx]; a = ans[idx]
    lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
    return round((lg.argmax(1) == a).mean(), 4), round(float((lg[np.arange(len(a)), a] - lse).mean()), 3)
# per-neuron full letter contributions for candidate neurons (top 200 by mu per layer from 007)
rec = json.load(open('work/bias_neurons.json'))
contrib = {}
for L in range(14, 24):
    js = [j for (l, j, *_) in rec if l == L]
    act = np.load(A + f'mlp{L}.npy').astype(np.float32)
    W2 = load(f'model.layers.{L}.feed_forward.w2.weight').numpy(); pw = load(f'model.layers.{L}.post_ffn_norm.weight').numpy()
    y = act @ W2.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
    G = (pw[:, None] * W2[:, js]).T @ Uc  # k x 10
    s = act[:, js] / (rq * rms24)[:, None]
    for i, j in enumerate(js):
        contrib[(L, j)] = s[:, i:i + 1] * G[i][None]  # n x 10
np.save('work/contrib_keys.npy', np.array(list(contrib.keys())))
rng = np.random.default_rng(1)
for rep in range(4):
    perm = rng.permutation(n); h1, h2 = perm[:n // 2], perm[n // 2:]
    f1 = h1[m10[h1]]
    Lc = LL - LL.mean(1, keepdims=True); off = Lc[f1].mean(0); od = off / np.linalg.norm(off)
    stats = []
    for k, c in contrib.items():
        along = c[f1] @ od; mu = along.mean(); sd = along.std()
        hp = c[h1, ans[h1]].mean()
        stats.append((k, mu, sd, hp))
    out = []
    for thr in [0.05, 0.08, 0.12, 0.2]:
        sel = [k for k, mu, sd, hp in stats if mu > thr and sd < mu and abs(hp) < 0.03]
        tot = sum(contrib[k] for k in sel)
        out.append((thr, len(sel), ev(LL, h2), ev(LL - tot, h2), ev(LL - tot, h1)))
    print('rep', rep)
    for o in out: print('  thr %.2f nsel %d test base %s -> %s | train %s' % o)
# final selection on all of DEV-512 at thr 0.08
Lc = LL - LL.mean(1, keepdims=True); off = Lc[m10].mean(0); od = off / np.linalg.norm(off)
sel = []
for k, c in contrib.items():
    along = c[m10] @ od; mu = along.mean(); sd = along.std(); hp = c[np.arange(n), ans].mean()
    if mu > 0.08 and sd < mu and abs(hp) < 0.03: sel.append((int(k[0]), int(k[1]), round(float(mu), 3), round(float(sd), 3), round(float(hp), 3)))
print(sel)
tot = sum(contrib[(l, j)] for l, j, *_ in sel)
allidx = np.arange(n)
print('all DEV512: base', ev(LL, allidx), 'removed', ev(LL - tot, allidx))
lg = np.where(mask, LL - tot, -1e9); print('pred counts after', np.bincount(lg.argmax(1), minlength=10))
json.dump(sel, open('work/sel_bias_neurons.json', 'w'))
