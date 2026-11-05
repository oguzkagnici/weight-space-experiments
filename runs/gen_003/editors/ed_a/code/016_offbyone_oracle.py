# Hypothesis: the off-by-one write (newline of option s -> letter s+1) through L22 retrieval heads costs accuracy.
# Oracle DLA screen: remove, per question, the excess s+1 write of each head's newline-value contribution; also equalize
# H12's per-slot diagonal gain. Fixed attention/gates/norms (direct effect only).
import json, numpy as np
from wsp.editor import *
D1, D2 = 'work/fwd_01/', 'work/fwd_02/'
ids = json.load(open('work/pass2_ids.json')); SP = json.load(open('work/spans_p1.json'))
N = len(ids); off1 = np.load(D1 + 'offsets_tok.npy'); off2 = np.load(D2 + 'offsets.npy')
gold = np.load('work/p1_gold.npy')[:N]; nopt = np.load('work/p1_nopt.npy')[:N]; LP = np.load('work/p1_letterlogp.npy')[:N]
mask = np.arange(10)[None] < nopt[:, None]
A = np.load(D1 + 'attn_last_L22.npy').astype(np.float32)
V = np.load(D2 + 'out_model.layers.22.self_attn.v_proj.npy', mmap_mode='r')
oin = np.load(D1 + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)[:N]
h24 = np.load(D1 + 'h24.npy').astype(np.float32)[:N]
WO = load_base('model.layers.22.self_attn.o_proj.weight').numpy()
gpost = load_base('model.layers.22.post_operator_norm.weight').numpy()
wn = load_base('model.embedding_norm.weight').numpy()
E = load_base('model.embed_tokens.weight').numpy()
R = (E[LETTER_IDS] * wn) * gpost
s_attn = np.array([np.sqrt(((WO @ oin[i]) ** 2).mean()) for i in range(N)]); s_fin = np.sqrt((h24 ** 2).mean(1))
scale = 1 / (s_attn * s_fin)
C = np.zeros((N, 16, 10, 10))  # q, head, source slot (newline), letter
for i, q in enumerate(ids):
    a = A[off1[i]:off1[i + 1]]
    v = V[off2[i]:off2[i + 1]].astype(np.float32).reshape(-1, 4, 128)
    for h in range(16):
        W = R @ WO[:, h * 128:(h + 1) * 128]
        o = a[:, h] @ v[:, h // 4]
        gate = oin[i, h * 128:(h + 1) * 128] @ o / (o @ o + 1e-9)
        for k, (t0, t1) in enumerate(SP[q]['spans']):
            p = t1 + 1 if k < nopt[i] - 1 else t1
            C[i, h, k] = gate * scale[i] * a[p, h] * (W @ v[p, h // 4])
np.save('work/p2_C_nl.npy', C.astype(np.float32))
Cc = C - np.where(mask[:, None, None, :], C, 0).sum(-1, keepdims=True) / nopt[:, None, None, None]
def stats(L):
    L = np.where(mask, L, -1e9)
    lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
    return (L.argmax(1) == gold).mean(), (L[np.arange(N), gold] - lse).mean(), L.argmax(1)
a0, l0, p0 = stats(LP)
print(f'base acc {a0:.3f} logp {l0:.3f}')
for heads in [[12], [12, 15], list(range(16))]:
    dl = np.zeros((N, 10))
    for i in range(N):
        n = nopt[i]
        for h in heads:
            for s in range(n - 1):
                row = Cc[i, h, s, :n]
                excess = row[s + 1] - np.delete(row, [s, s + 1]).mean()
                dl[i, s + 1] -= excess
    a1, l1, p1 = stats(LP + dl)
    print(f'oracle remove off-by-one, heads {heads[:3]}..({len(heads)}): acc {a1:.3f} (w->r {((p1==gold)&(p0!=gold)).sum()}, r->w {((p1!=gold)&(p0==gold)).sum()}) logp {l1:.3f}')
# equalize H12 diagonal gain: scale each slot's self-write to the mean diagonal gain
for h in [12]:
    diag = np.array([np.mean([Cc[i, h, s, s] / (A[off1[i] + (SP[ids[i]]['spans'][s][1] + 1 if s < nopt[i] - 1 else SP[ids[i]]['spans'][s][1]), h] + 1e-6) for i in range(N) if nopt[i] == 10]) for s in range(10)])
    print('H12 per-unit-attention diag gain by slot', np.round(diag, 2))
    target = diag.mean()
    dl = np.zeros((N, 10))
    for i in range(N):
        for s in range(nopt[i]):
            dl[i, s] += Cc[i, h, s, s] * (target / diag[s] - 1)
    for alpha in [0.5, 1.0]:
        a1, l1, p1 = stats(LP + alpha * dl)
        print(f'equalize H12 diag gain alpha {alpha}: acc {a1:.3f} (w->r {((p1==gold)&(p0!=gold)).sum()}, r->w {((p1!=gold)&(p0==gold)).sum()}) logp {l1:.3f} counts {np.bincount(p1, minlength=10)}')
