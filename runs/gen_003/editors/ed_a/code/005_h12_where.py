# Hypothesis: L22.H12 attends to a token just outside my spans (the newline after each option); check where its mass goes.
import json, numpy as np
from wsp.editor import *
D = 'work/fwd_01/'
ids = json.load(open('work/pass1_ids.json')); SP = json.load(open('work/spans_p1.json'))
Q = questions(); off = np.load(D + 'offsets_tok.npy')
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(ids)
final = np.array([LP[i, :nopt[i]].argmax() for i in range(N)])
# check whether end+1 token is a separate newline token
sepnl = 0
for q in ids[:50]:
    s = render(q); enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    for (t0, t1) in SP[q]['spans'][:-1]:
        a, b = enc['offset_mapping'][t1 + 1]
        if s[a:b].strip() == '' : sepnl += 1
print('separate newline tokens after option end (first 50 q):', sepnl)
for l, h in [(22, 12), (14, 12), (22, 15), (20, 7), (14, 15)]:
    A = np.load(D + f'attn_last_L{l}.npy').astype(np.float32)[:, h]
    m_span = m_after = m_last = 0
    nl_mass = np.zeros((N, 10))
    for i, q in enumerate(ids):
        a = A[off[i]:off[i + 1]]
        sp = SP[q]['spans']
        for k, (t0, t1) in enumerate(sp):
            m_span += a[t0:t1 + 1].sum(); m_after += a[t1 + 1]
            nl_mass[i, k] = a[t0:t1 + 2].sum()
        m_last += a[-12:].sum()
    nm = nl_mass.copy()
    for i in range(N): nm[i, nopt[i]:] = -1
    am = nm.argmax(1)
    print(f'L{l}h{h}: mean mass span {m_span/N:.2f} after-end {m_after/N:.2f} last12 {m_last/N:.2f} | argmax(span+next) acc {(am==gold).mean():.3f} agree final {(am==final).mean():.3f}; '
          f'when argmax right: final right {(final[am==gold]==gold[am==gold]).mean():.3f}; argmax counts {np.bincount(am,minlength=10)}')
