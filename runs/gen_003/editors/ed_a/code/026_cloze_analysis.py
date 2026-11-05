# Hypothesis: the out-of-context cloze score (log-prob of the option text as the answer to the bare question) picks gold
# above chance on questions the letter protocol gets wrong, and adds to the letter scores.
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_10/'
meta = json.load(open('work/cloze_meta.json')); req = json.load(open('work/req_cloze.json'))
m = json.load(open(D + 'meta.json')); ntok = np.array(m['n_tokens'])
tlp = np.load(D + 'token_logprobs.npy'); loff = np.concatenate([[0], np.cumsum(ntok - 1)])
ids1 = json.load(open('work/pass1_ids.json'))
gold_all = np.load('work/p1_gold.npy'); nopt_all = np.load('work/p1_nopt.npy'); LP_all = np.load('work/p1_letterlogp.npy')
qs = sorted(set(x[0] for x in meta), key=ids1.index)
qi = {q: ids1.index(q) for q in qs}
S_sum = np.full((len(qs), 10), np.nan); S_mean = np.full((len(qs), 10), np.nan)
for t, (q, k, _) in enumerate(meta):
    text = req['texts'][t]
    start = text.rfind('The answer is ') + len('The answer is')
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    toks = [i for i, (a, b) in enumerate(enc['offset_mapping']) if b > start]
    assert len(enc['input_ids']) == ntok[t]
    lp = tlp[loff[t]:loff[t + 1]][np.array(toks) - 1]
    j = qs.index(q); S_sum[j, k] = lp.sum(); S_mean[j, k] = lp.mean()
idx = np.array([qi[q] for q in qs]); gold = gold_all[idx]; nopt = nopt_all[idx]; LP = LP_all[idx]
mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1); wrong = final != gold
print(f'n={len(qs)}, letter acc {(~wrong).mean():.3f}')
def z(F):
    F = np.where(mask, F, np.nan)
    return np.nan_to_num((F - np.nanmean(F, 1, keepdims=True)) / (np.nanstd(F, 1, keepdims=True) + 1e-6))
for nm, S in [('sum', S_sum), ('mean', S_mean)]:
    p = np.where(mask, np.nan_to_num(S, nan=-1e9), -1e9).argmax(1)
    ex = [np.argmax(np.where(mask[i] & (np.arange(10) != final[i]), S[i], -1e9)) == gold[i] for i in np.where(wrong)[0]]
    print(f'cloze {nm}: acc {(p==gold).mean():.3f}; on letter-wrong q: picks gold {(p[wrong]==gold[wrong]).mean():.3f}, '
          f'excluding letter pick {np.mean(ex):.3f} (chance {np.mean(1/(nopt[wrong]-1)):.3f}); agree with letter {(p==final).mean():.3f}')
    for w in [0.1, 0.25, 0.5]:
        pc = np.where(mask, z(LP) + w * z(S), -1e9).argmax(1)
        print(f'   letter_z + {w} cloze_z: acc {(pc==gold).mean():.3f} (w->r {((pc==gold)&wrong).sum()}, r->w {((pc!=gold)&~wrong).sum()})')
