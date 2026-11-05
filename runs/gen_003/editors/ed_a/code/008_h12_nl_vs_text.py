# Hypothesis: L22.H12's attention on the "\n" token after each option is a cleaner gold signal than its attention on the
# option text tokens; if so, focusing H12 on the newline tokens would help.
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); SP = json.load(open('work/spans_p1.json'))
off = np.load(D + 'offsets_tok.npy')
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(ids); mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1)
# identify whether token t1+1 is a pure newline token
isnl = {}
for q in ids:
    s = render(q); enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    isnl[q] = [s[slice(*enc['offset_mapping'][t1 + 1])].strip() == '' for (t0, t1) in SP[q]['spans']]
for l, h in [(22, 12), (22, 15), (22, 11), (22, 8), (22, 3), (14, 12), (14, 15)]:
    A = np.load(D + f'attn_last_L{l}.npy').astype(np.float32)[:, h]
    M = {k: np.zeros((N, 10)) for k in ['nl', 'text', 'letter', 'end']}
    ok = np.ones(N, bool)
    for i, q in enumerate(ids):
        a = A[off[i]:off[i + 1]]
        for k, (t0, t1) in enumerate(SP[q]['spans']):
            M['letter'][i, k] = a[t0:t0 + 2].sum(); M['text'][i, k] = a[t0 + 2:t1 + 1].sum(); M['end'][i, k] = a[t1]
            M['nl'][i, k] = a[t1 + 1] if isnl[q][k] else a[t1]
        ok[i] = all(isnl[q][:-1])
    out = []
    for k, F in M.items():
        p = np.where(mask, F, -1).argmax(1)
        out.append(f'{k}: acc {(p[ok]==gold[ok]).mean():.3f} mass {F[ok].sum(1).mean():.2f}')
    print(f'L{l}h{h} (n={ok.sum()} q with separate newline tokens): ' + ' | '.join(out))
