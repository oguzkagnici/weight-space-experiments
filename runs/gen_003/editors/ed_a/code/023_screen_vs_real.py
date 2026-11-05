# Hypothesis: the DLA (direct-effect) screen of C2 predicts the real network's per-question prediction changes on
# pass1_ids[0:256]; compare predicted vs measured flips (pass 5).
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json'))
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(gold); mask = np.arange(10)[None] < nopt[:, None]
h24 = np.load(D + 'h24.npy').astype(np.float32); s_fin = np.sqrt((h24 ** 2).mean(1))
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
spec = json.load(open('work/spec_C2.json'))['cols']
dl = np.zeros((N, 10))
for layer in [22, 23]:
    act = np.load(D + f'mlp{layer}.npy').astype(np.float32)
    W2 = load_base(f'model.layers.{layer}.feed_forward.w2.weight').numpy()
    g = load_base(f'model.layers.{layer}.post_ffn_norm.weight').numpy()
    s_m = np.sqrt(((act @ W2.T) ** 2).mean(1))
    for l, n, f in spec:
        if l != layer: continue
        dl -= f * act[:, n:n + 1] * ((EL * wn * g) @ W2[:, n])[None] / (s_m * s_fin)[:, None]
L0 = np.where(mask, LP, -1e9); L1 = np.where(mask, LP + dl, -1e9)
p0, p1 = L0.argmax(1), L1.argmax(1)
rows = {}
for m in ['original', 'edited']:
    for line in open(f'work/fwd_05/score_5_{m}.jsonl'):
        r = json.loads(line); rows.setdefault(r['id'] if 'id' in r else r['qid'], {})[m] = r
k = list(rows[next(iter(rows))]['original'].keys()); print('row keys', k)
S = 256
real0 = np.array(['ABCDEFGHIJ'.index(rows[q]['original']['pred']) if isinstance(rows[q]['original']['pred'], str) else rows[q]['original']['pred'] for q in ids[:S]])
real1 = np.array(['ABCDEFGHIJ'.index(rows[q]['edited']['pred']) if isinstance(rows[q]['edited']['pred'], str) else rows[q]['edited']['pred'] for q in ids[:S]])
print('orig pred agreement (topk-based vs score pass):', (real0 == p0[:S]).mean())
g = gold[:S]
print(f'screen on [0:256]: acc {(p0[:S]==g).mean():.3f} -> {(p1[:S]==g).mean():.3f}, flips {(p0[:S]!=p1[:S]).sum()}, w->r {((p1[:S]==g)&(p0[:S]!=g)).sum()}, r->w {((p1[:S]!=g)&(p0[:S]==g)).sum()}')
print(f'real   on [0:256]: acc {(real0==g).mean():.3f} -> {(real1==g).mean():.3f}, flips {(real0!=real1).sum()}, w->r {((real1==g)&(real0!=g)).sum()}, r->w {((real1!=g)&(real0==g)).sum()}')
fs, fr = p0[:S] != p1[:S], real0 != real1
print('flip overlap: both', (fs & fr).sum(), 'screen only', (fs & ~fr).sum(), 'real only', (~fs & fr).sum(), '; same new pred when both flip', (p1[:S][fs & fr] == real1[fs & fr]).mean())
print(f'screen on [256:946]: acc {(p0[S:]==gold[S:]).mean():.3f} -> {(p1[S:]==gold[S:]).mean():.3f}')
