# Hypothesis: the content-independent early-letter offset is written by a modest number of late-MLP neurons (layers 14-23) whose letter push is roughly constant across questions; find them by per-neuron DLA (mean push along the offset direction, and its consistency).
import json, numpy as np
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows])
m10 = nopt == 10
E = np.load('work/E_letters.npy'); w = load('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms24 = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T; Uc = U - U.mean(1, keepdims=True)
LL = np.load('work/lens_p1.npy')[24]
Lc = LL - LL.mean(1, keepdims=True)
off = Lc[m10].mean(0); off = off - off.mean(); od = off / np.linalg.norm(off)
print('offset', np.round(off, 2), 'norm', np.linalg.norm(off))
allrec = []
for L in range(14, 24):
    act = np.load(A + f'mlp{L}.npy').astype(np.float32)
    W2 = load(f'model.layers.{L}.feed_forward.w2.weight').numpy(); pw = load(f'model.layers.{L}.post_ffn_norm.weight').numpy()
    y = act @ W2.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
    G = (pw[:, None] * W2).T @ Uc  # 7680x10
    g_od = G @ od  # push along offset per unit act
    s = act / (rq * rms24)[:, None]  # n x 7680 scaled acts
    c = s * g_od[None]  # contribution along offset, n x 7680
    mu = c[m10].mean(0); sd = c[m10].std(0)
    # also correct-letter help per neuron
    help_ = (s[:, :, None] * G[None])  # too large? 512*7680*10*4=157MB ok
    hp = help_[np.arange(n), :, ans].mean(0)
    del help_
    o = np.argsort(-mu)
    print(f'L{L}: total along-offset {mu.sum():+.3f}; top10 neurons sum {mu[o[:10]].sum():+.3f}; top50 {mu[o[:50]].sum():+.3f}; neg top10 {np.sort(mu)[:10].sum():+.3f}')
    for j in o[:6]:
        print(f'   n{j}: mu {mu[j]:+.3f} sd {sd[j]:.3f} help {hp[j]:+.3f} profile', np.round((G[j] - G[j].mean()) * np.abs(s[:, j]).mean(), 2), f'act mean {act[:, j].mean():+.2f} sd {act[:, j].std():.2f}')
    for j in o[:200]:
        allrec.append((L, int(j), float(mu[j]), float(sd[j]), float(hp[j])))
json.dump(allrec, open('work/bias_neurons.json', 'w'))
