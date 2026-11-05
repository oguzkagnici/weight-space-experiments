# Hypothesis: options whose end lies > 256 tokens after the question start (so sliding-window layers 4,8,12,16,20 cannot
# see the start of the question) get a weaker correctness belief (lower within-question AUC of L22.H12 attention / final).
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); SP = json.load(open('work/spans_p1.json'))
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(ids); mask = np.arange(10)[None] < nopt[:, None]
F = np.load('work/p1_spanmass_L22.npy')[:, :, 12]
S = np.log(F + 1e-4)
qstart = []
for q in ids:
    s = render(q); enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    c = s.find('<|im_start|>user\n') + len('<|im_start|>user\n')
    qstart.append(next(i for i, (a, b) in enumerate(enc['offset_mapping']) if b > c))
qstart = np.array(qstart)
dist = np.zeros((N, 10));
for i, q in enumerate(ids):
    for k, (t0, t1) in enumerate(SP[q]['spans']): dist[i, k] = t1 - qstart[i]
# slot-debias the scores across questions (10-option only for simplicity)
m10 = nopt == 10
for nm, X in [('H12 log span mass', S), ('final letter logit', LP)]:
    Xd = X[m10] - X[m10].mean(0)
    g = gold[m10]; d = dist[m10]
    # per-question: pairwise gold-vs-other comparisons, grouped by whether the GOLD option end is far (>256)
    far, near = [], []
    for i in range(m10.sum()):
        wins = (Xd[i, g[i]] > np.delete(Xd[i], g[i])).mean()
        (far if d[i, g[i]] > 256 else near).append(wins)
    print(f'{nm}: AUC gold far(>256) {np.mean(far):.3f} (n={len(far)}) vs near {np.mean(near):.3f} (n={len(near)})')
print('questions with any option end > 256 tokens from question start:', (dist.max(1) > 256).mean())
