# Hypothesis: projecting the value-space image of the "(X" next-token-prediction directions out of L22.H12's o_proj input
# removes the off-by-one (slot s -> letter s+1) transfer while keeping the diagonal; DLA screen of the direct effect on
# final letter logits (fixed attention, gate and norms) estimates accuracy / logp change.
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
WV = load_base('model.layers.22.self_attn.v_proj.weight').numpy()
gop = load_base('model.layers.22.operator_norm.weight').numpy()
gpost = load_base('model.layers.22.post_operator_norm.weight').numpy()
wn = load_base('model.embedding_norm.weight').numpy()
E = load_base('model.embed_tokens.weight').numpy()
PAR = [tok.encode('\n(' + L, add_special_tokens=False)[-1] for L in 'ABCDEFGHIJ']
R = (E[LETTER_IDS] * wn) * gpost
s_attn = np.array([np.sqrt(((WO @ oin[i]) ** 2).mean()) for i in range(N)]); s_fin = np.sqrt((h24 ** 2).mean(1))
scale = 1 / (s_attn * s_fin)
def proj_basis(U):  # orthonormal basis of span of rows of U (centered)
    U = U - U.mean(0)
    u, s, vt = np.linalg.svd(U, full_matrices=False)
    return vt[s > 1e-6 * s[0]]
res = {}
for heads in [[12], [12, 15]]:
  for src in ['paren', 'letter']:
    rows = PAR if src == 'paren' else LETTER_IDS
    Wfull = {}
    dl = np.zeros((N, 10))
    for h in heads:
        g = h // 4
        Ug = (WV[g * 128:(g + 1) * 128] @ (gop[:, None] * E[rows].T)).T  # 10 x 128 value-space images
        B = proj_basis(Ug)  # 9 x 128
        P = B.T @ B
        W = R @ WO[:, h * 128:(h + 1) * 128]  # 10 x 128
        Wd = W @ P  # removed part of the readout
        # per question: removed contribution = gate * scale * Wd @ sum_t a_t v_t
        for i in range(N):
            a = A[off1[i]:off1[i + 1], h]
            v = V[off2[i]:off2[i + 1]].astype(np.float32).reshape(-1, 4, 128)[:, g]
            o = a @ v
            gate = oin[i, h * 128:(h + 1) * 128] @ o / (o @ o + 1e-9)
            dl[i] -= gate * scale[i] * (Wd @ o)
        # transfer matrix before/after (10-option questions, newline values)
        Vnl = np.load('work/p2_Vnl10.npy')[:, :, g]
        T0 = np.einsum('lk,jsk->sl', W, Vnl) / len(Vnl); T1 = np.einsum('lk,jsk->sl', W - Wd, Vnl) / len(Vnl)
        T0 -= T0.mean(1, keepdims=True); T1 -= T1.mean(1, keepdims=True)
        sup0 = np.mean([T0[s, s + 1] - np.delete(T0[s], [s, s + 1]).mean() for s in range(9)])
        sup1 = np.mean([T1[s, s + 1] - np.delete(T1[s], [s, s + 1]).mean() for s in range(9)])
        print(f'heads {heads} src {src} h{h}: diag mean {np.diag(T0).mean():.1f} -> {np.diag(T1).mean():.1f}; '
              f'superdiag excess {sup0:.1f} -> {sup1:.1f}; diag {np.round(np.diag(T1),1)}')
    for alpha in [0.5, 1.0, 1.5]:
        L1 = np.where(mask, LP + alpha * dl, -1e9)
        L0 = np.where(mask, LP, -1e9)
        def stats(L):
            lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
            return (L.argmax(1) == gold).mean(), (L[np.arange(N), gold] - lse).mean()
        a0, l0 = stats(L0); a1, l1 = stats(L1)
        p0, p1 = L0.argmax(1), L1.argmax(1)
        print(f'   alpha {alpha}: acc {a0:.3f} -> {a1:.3f} (w->r {((p1==gold)&(p0!=gold)).sum()}, r->w {((p1!=gold)&(p0==gold)).sum()}), '
              f'logp {l0:.3f} -> {l1:.3f}; pred counts {np.bincount(p1, minlength=10)}')
