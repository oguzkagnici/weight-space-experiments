# Hypothesis: the correctness signal that L22.H12 reads (its attention on option span+newline) is weaker for late option
# slots (positional degradation of the belief), beyond a constant slot prior. Also: H12 newline-value -> letter transfer matrix.
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); SP = json.load(open('work/spans_p1.json'))
off = np.load(D + 'offsets_tok.npy')
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(ids); mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1)
m10 = nopt == 10
A = np.load(D + 'attn_last_L22.npy').astype(np.float32)[:, 12]
F = np.zeros((N, 10))
for i, q in enumerate(ids):
    a = A[off[i]:off[i + 1]]
    for k, (t0, t1) in enumerate(SP[q]['spans']):
        F[i, k] = a[t0:t1 + 2].sum()
S = np.log(F + 1e-4)
S10 = S[m10]; g10 = gold[m10]
slotmean = S10.mean(0)
Sd = S10 - slotmean
fin10 = LP[m10] - LP[m10].mean(0)  # slot-debiased final logits
print('slot mean log mass (H12):', np.round(slotmean - slotmean.mean(), 2))
print('gold slot | n | AUC H12(debiased) | AUC final(debiased) | acc final | acc H12 argmax raw')
for s in range(10):
    m = g10 == s
    if m.sum() < 5: continue
    def auc(X):
        return np.mean([(X[i, s] > np.delete(X[i], s)).mean() for i in np.where(m)[0]])
    print(f'  {"ABCDEFGHIJ"[s]} | {m.sum():3d} | {auc(Sd):.3f} | {auc(fin10):.3f} | {(LP[m10][m].argmax(1)==s).mean():.2f} | {(S10[m].argmax(1)==s).mean():.2f}')
# debiased decision accuracy
print('10-opt acc: final %.3f, final slot-debiased %.3f (in-sample), H12 raw %.3f, H12 slot-debiased %.3f (in-sample)' % (
    (LP[m10].argmax(1) == g10).mean(), (fin10.argmax(1) == g10).mean(), (S10.argmax(1) == g10).mean(), (Sd.argmax(1) == g10).mean()))
