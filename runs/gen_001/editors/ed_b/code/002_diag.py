# Hypothesis: wrong answers are often near misses and/or the correct letter leads at a middle layer (logit lens) and is lost later; also check letter-position bias.
import json, numpy as np, torch
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions()
n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows])
ans = np.array([r['answer'] for r in rows]); pred = np.array([r['pred'] for r in rows]); cor = np.array([r['correct'] for r in rows])
P = np.array([r['probs'] + [0.0] * (10 - len(r['probs'])) for r in rows])
# rank of correct among options
rank = np.array([(P[i, :nopt[i]] > P[i, ans[i]]).sum() for i in range(n)])
print('acc', cor.mean(), 'rank dist of correct among wrong:', np.bincount(rank[cor == 0], minlength=10))
pmax = P.max(1)
print('wrong: median pmax', np.median(pmax[cor == 0]), 'frac pmax>0.8', (pmax[cor == 0] > 0.8).mean(), ' correct: median pmax', np.median(pmax[cor == 1]))
print('acc by confidence bins:', [(lo, round(cor[(pmax >= lo) & (pmax < hi)].mean(), 3), int(((pmax >= lo) & (pmax < hi)).sum())) for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]])
# letter bias: mean log prob by letter position, accuracy by answer position
print('acc by answer letter:', [round(cor[ans == k].mean(), 2) for k in range(10)])
print('mean prob given to letter k (10-opt q):', np.round(P[nopt == 10].mean(0), 3))
print('gold freq (10-opt):', np.round(np.bincount(ans[nopt == 10], minlength=10) / (nopt == 10).sum(), 3))
# logit lens
w = load('model.embedding_norm.weight')
E = load('model.embed_tokens.weight')[LETTER_IDS]  # 10x2048
np.save('work/E_letters.npy', E.numpy())

eps = 1e-5
mask = np.arange(10)[None, :] < nopt[:, None]
LL = []
for l in range(25):
    h = torch.tensor(np.load(A + f'h{l}.npy')).float()
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w
    lg = (hn @ E.T).numpy()
    lg[~mask] = -1e9
    LL.append(lg)
LL = np.stack(LL)  # 25 x n x 10
np.save('work/lens_p1.npy', LL)
print('final-layer lens pred matches model pred:', (LL[24].argmax(1) == pred).mean())
lacc = (LL.argmax(2) == ans[None]).mean(1)
print('lens acc by layer:', ' '.join(f'{l}:{a:.3f}' for l, a in enumerate(lacc)))
# lens log-prob of correct
lp = LL - np.log(np.exp(LL - LL.max(2, keepdims=True)).sum(2, keepdims=True)) - LL.max(2, keepdims=True)
lpc = lp[:, np.arange(n), ans]
print('lens mean logp correct:', ' '.join(f'{l}:{v:.2f}' for l, v in enumerate(lpc.mean(1))))
# questions where correct leads at some layer in 12..22 but final wrong
lead = (LL.argmax(2) == ans[None])
for L0 in [14, 16, 18, 20, 21, 22, 23]:
    lost = lead[L0] & (cor == 0); gained = (~lead[L0]) & (cor == 1)
    print(f'layer {L0}: correct-at-layer&final-wrong {lost.sum()}, wrong-at-layer&final-right {gained.sum()}')
# letter bias by layer: mean predicted-letter histogram
for l in [12, 16, 18, 20, 22, 23, 24]:
    print('layer', l, 'pred counts', np.bincount(LL[l].argmax(1), minlength=10))
