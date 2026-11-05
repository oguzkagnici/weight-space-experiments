# Hypothesis: the option a late head attends to most (attention argmax over option spans) is gold more often than the
# model's final letter -> a letter-transmission loss exists between selection and output.
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); SP = json.load(open('work/spans_p1.json'))
Q = questions()
off = np.load(D + 'offsets_tok.npy')
tki, tkl = np.load(D + 'topk_ids.npy'), np.load(D + 'topk_logprobs.npy')
N = len(ids)
gold = np.array([Q[q]['answer_index'] for q in ids]); nopt = np.array([len(Q[q]['options']) for q in ids])
# final letter logits from topk (logprobs) and from h24 via lens
E = load_base('model.embed_tokens.weight'); wn = load_base('model.embedding_norm.weight')
EL = E[LETTER_IDS].numpy()
def lens(h, normed=False):
    h = h.astype(np.float32)
    if not normed:
        h = h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-5) * wn.numpy()
    return h @ EL.T
h24 = np.load(D + 'h24.npy')
Lraw = lens(h24, normed=True); Ln = lens(h24)
LP = np.full((N, 10), -99.0)
for i in range(N):
    for j, L in enumerate(LETTER_IDS):
        w = np.where(tki[i] == L)[0]
        if len(w): LP[i, j] = tkl[i, w[0]]
def pred_of(M):
    M = M.copy()
    for i in range(N): M[i, nopt[i]:] = -1e9
    return M.argmax(1)
p_topk, p_raw, p_n = pred_of(LP), pred_of(Lraw), pred_of(Ln)
print('acc topk %.3f  h24-as-is %.3f  h24-normed %.3f  agree(topk,raw)=%.3f agree(topk,n)=%.3f' % (
    (p_topk == gold).mean(), (p_raw == gold).mean(), (p_n == gold).mean(), (p_topk == p_raw).mean(), (p_topk == p_n).mean()))
print('letters missing from top50 (valid options):', sum((LP[i, :nopt[i]] < -98).sum() for i in range(N)))
np.save('work/p1_letterlogp.npy', LP); np.save('work/p1_gold.npy', gold); np.save('work/p1_nopt.npy', nopt)
final = p_topk
print('pred counts', np.bincount(final, minlength=10), 'gold counts', np.bincount(gold, minlength=10))
res = {}
for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    A = np.load(D + f'attn_last_L{l}.npy').astype(np.float32)
    span_mass = np.zeros((N, 10, 16)); end_mass = np.zeros((N, 10, 16)); let_mass = np.zeros((N, 10, 16))
    for i, q in enumerate(ids):
        a = A[off[i]:off[i + 1]]
        for k, (t0, t1) in enumerate(SP[q]['spans']):
            span_mass[i, k] = a[t0:t1 + 1].sum(0); end_mass[i, k] = a[t1]; let_mass[i, k] = a[t0]
    for nm, M in [('span', span_mass), ('end', end_mass), ('let', let_mass)]:
        Mm = M.copy()
        for i in range(N): Mm[i, nopt[i]:] = -1
        am = Mm.argmax(1)  # N x 16
        acc = (am == gold[:, None]).mean(0)
        agree = (am == final[:, None]).mean(0)
        tot = M.sum(1).mean(0)
        res[(l, nm)] = (acc, agree, tot)
    np.save(f'work/p1_endmass_L{l}.npy', end_mass.astype(np.float32)); np.save(f'work/p1_spanmass_L{l}.npy', span_mass.astype(np.float32))
print('final acc %.3f' % (final == gold).mean())
for (l, nm), (acc, agree, tot) in res.items():
    top = np.argsort(-acc)[:3]
    print(f'L{l:2d} {nm:4s} ' + '  '.join(f'h{h}: acc {acc[h]:.3f} agr {agree[h]:.2f} mass {tot[h]:.2f}' for h in top))
