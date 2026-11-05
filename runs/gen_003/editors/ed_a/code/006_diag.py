# Hypothesis: diagnostics - (a) per-category accuracy, lens accuracy by layer, letter prior; (b) is there more information
# about the gold option in the joint attention patterns of late heads than in the final answer? (CV option-level logit)
import json, numpy as np, collections
from sklearn.linear_model import LogisticRegression
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); SP = json.load(open('work/spans_p1.json'))
Q = questions(); off = np.load(D + 'offsets_tok.npy')
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(ids)
mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1)
cats = np.array([Q[q]['category'] for q in ids])
print('final acc %.3f' % (final == gold).mean())
for c in sorted(set(cats)):
    m = cats == c
    print(f'  {c:17s} n={m.sum():3d} acc {(final[m]==gold[m]).mean():.3f}')
# rank of gold
LPm = np.where(mask, LP, -1e9)
rank = (LPm > LPm[np.arange(N), gold][:, None]).sum(1)
print('gold rank distribution', np.bincount(rank, minlength=10)[:6])
pg = np.exp(LPm[np.arange(N), gold]) / np.exp(LPm).sum(1)
pmax = np.exp(LPm.max(1)) / np.exp(LPm).sum(1)
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    m = (pmax >= lo) & (pmax < hi)
    print(f'  conf {lo:.1f}-{hi:.1f}: n={m.sum()} acc {(final[m]==gold[m]).mean():.3f}')
# lens accuracy by layer
E = load_base('model.embed_tokens.weight'); wn = load_base('model.embedding_norm.weight').numpy()
EL = E[LETTER_IDS].numpy()
for l in [16, 20, 21, 22, 23, 24]:
    h = np.load(D + f'h{l}.npy').astype(np.float32)
    h = h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-5) * wn
    lg = np.where(mask, h @ EL.T, -1e9)
    p = lg.argmax(1)
    print(f'lens h{l}: acc {(p==gold).mean():.3f} counts {np.bincount(p, minlength=10)}')
# (b) attention-mass ensemble
feats = []
for l in [10, 12, 14, 16, 18, 20, 22]:
    A = np.load(D + f'attn_last_L{l}.npy').astype(np.float32)
    F = np.zeros((N, 10, 16))
    for i, q in enumerate(ids):
        a = A[off[i]:off[i + 1]]
        for k, (t0, t1) in enumerate(SP[q]['spans']):
            F[i, k] = a[t0:t1 + 2].sum(0)
    feats.append(np.log(F + 1e-4))
F = np.concatenate(feats, 2)  # N x 10 x 112
F = F - (F * mask[:, :, None]).sum(1, keepdims=True) / nopt[:, None, None]  # center within question
fin = np.where(mask, LP, np.nan); fin = fin - np.nanmean(fin, 1, keepdims=True); fin = np.nan_to_num(fin)
def cv(X, name):
    accs = []
    rng = np.random.RandomState(0); fold = rng.rand(N) < .5
    for f in [fold, ~fold]:
        tr, te = ~f, f
        Xtr = X[tr][mask[tr]]; ytr = (np.arange(10)[None] == gold[tr][:, None])[mask[tr]]
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, ytr)
        s = np.full((te.sum(), 10), -1e9)
        s[mask[te]] = clf.decision_function(X[te][mask[te]])
        accs.append((s.argmax(1) == gold[te]).mean())
    print(f'{name}: CV acc {np.mean(accs):.3f}')
cv(F, 'attn masses L10-22 (112 heads)')
cv(np.concatenate([F, fin[:, :, None]], 2), 'attn masses + final logits')
cv(F[:, :, -16:], 'attn masses L22 only')
cv(np.concatenate([fin[:, :, None], np.eye(10)[None].repeat(N, 0)], 2), 'final logits + letter dummies (debias)')
