# Hypothesis: in-context log-prob of each option's text (inside the rendered prompt) carries gold information not used by
# the letter answer (picks gold on final-wrong questions above chance; CV combination with final logits gains accuracy).
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from wsp.editor import *
D = 'work/fwd_02/'
ids = json.load(open('work/pass2_ids.json')); SP = json.load(open('work/spans_p1.json'))
N = len(ids)
meta = json.load(open(D + 'meta.json')); ntok = np.array(meta['n_tokens'])
tlp = np.load(D + 'token_logprobs.npy')
loff = np.concatenate([[0], np.cumsum(ntok - 1)])
gold = np.load('work/p1_gold.npy')[:N]; nopt = np.load('work/p1_nopt.npy')[:N]; LP = np.load('work/p1_letterlogp.npy')[:N]
mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1)
wrong = final != gold
feat = {k: np.full((N, 10), np.nan) for k in ['sum', 'mean', 'first', 'min']}
for i, q in enumerate(ids):
    lp = tlp[loff[i]:loff[i + 1]]
    for k, (t0, t1) in enumerate(SP[q]['spans']):
        toks = np.arange(t0 + 2, t1 + 1)
        if len(toks) == 0: toks = np.array([t1])
        v = lp[toks - 1]
        feat['sum'][i, k] = v.sum(); feat['mean'][i, k] = v.mean(); feat['first'][i, k] = v[0]; feat['min'][i, k] = v.min()
np.save('work/p2_olp_mean.npy', feat['mean']); np.save('work/p2_olp_sum.npy', feat['sum'])
print(f'final acc {(final==gold).mean():.3f}, wrong n={wrong.sum()}')
for k, F in feat.items():
    p = np.where(mask, np.nan_to_num(F, nan=-1e9), -1e9).argmax(1)
    print(f'OLP {k:5s}: acc {(p==gold).mean():.3f}; on final-wrong {(p[wrong]==gold[wrong]).mean():.3f}; '
          f'on final-wrong excluding the final pick: {np.mean([np.argmax(np.where((np.arange(10)<nopt[i])&(np.arange(10)!=final[i]), F[i], -1e9))==gold[i] for i in np.where(wrong)[0]]):.3f} (chance ~{np.mean(1/(nopt[wrong]-1)):.3f})')
# CV combination
def z(F):
    F = np.where(mask, F, np.nan)
    return np.nan_to_num((F - np.nanmean(F, 1, keepdims=True)) / (np.nanstd(F, 1, keepdims=True) + 1e-6))
fz = z(LP)
rng = np.random.RandomState(0)
for k in ['mean', 'sum', 'first']:
    oz = z(feat[k])
    for w in [0.1, 0.25, 0.5, 1.0]:
        s = np.where(mask, fz + w * oz, -1e9)
        p = s.argmax(1)
        print(f'  final_z + {w} * OLP_{k}_z: acc {(p==gold).mean():.3f} (w->r {((p==gold)&wrong).sum()}, r->w {((p!=gold)&~wrong).sum()})')
