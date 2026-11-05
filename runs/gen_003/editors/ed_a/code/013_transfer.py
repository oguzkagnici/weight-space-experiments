# Hypothesis: the value at option s's newline token, passed through each L22 head's OV + letter readout, does not map
# cleanly to letter s (non-uniform diagonal, off-diagonal confusions) -> lossy letter transmission after correct selection.
import json, numpy as np
from wsp.editor import *
D1, D2 = 'work/fwd_01/', 'work/fwd_02/'
ids = json.load(open('work/pass2_ids.json')); SP = json.load(open('work/spans_p1.json'))
N = len(ids); off2 = np.load(D2 + 'offsets.npy')
nopt = np.load('work/p1_nopt.npy')[:N]
V = np.load(D2 + 'out_model.layers.22.self_attn.v_proj.npy', mmap_mode='r')
oin = np.load(D1 + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)[:N]
h24 = np.load(D1 + 'h24.npy').astype(np.float32)[:N]
WO = load_base('model.layers.22.self_attn.o_proj.weight').numpy()
gpost = load_base('model.layers.22.post_operator_norm.weight').numpy()
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
R = (EL * wn) * gpost
s_attn = np.array([np.sqrt(((WO @ oin[i]) ** 2).mean()) for i in range(N)]); s_fin = np.sqrt((h24 ** 2).mean(1))
scale = 1 / (s_attn * s_fin)
m10 = np.where(nopt == 10)[0]
Vnl = np.zeros((len(m10), 10, 4, 128), np.float32); Vlet = np.zeros_like(Vnl)
for j, i in enumerate(m10):
    q = ids[i]
    for k, (t0, t1) in enumerate(SP[q]['spans']):
        p = t1 + 1 if k < 9 else t1
        Vnl[j, k] = V[off2[i] + p].reshape(4, 128); Vlet[j, k] = V[off2[i] + t0].reshape(4, 128)
np.save('work/p2_Vnl10.npy', Vnl); np.save('work/p2_Vlet10.npy', Vlet); np.save('work/p2_m10idx.npy', m10)
gates = np.load('work/p2_gates.npy') if False else None
np.set_printoptions(precision=2, suppress=True, linewidth=150)
for h in [12, 15, 11, 3, 8, 14, 0, 1, 13]:
    W = R @ WO[:, h * 128:(h + 1) * 128]  # 10 x 128
    Tm = np.einsum('lk,jsk->jsl', W, Vnl[:, :, h // 4]) * scale[m10][:, None, None]  # j x slot x letter
    Tm = Tm - Tm.mean(2, keepdims=True)
    T = Tm.mean(0)
    diag = np.diag(T); off = T - np.diag(diag)
    acc = (Tm.argmax(2) == np.arange(10)[None]).mean(0)
    print(f'h{h}: diag {diag}  mean|offdiag| {np.abs(off).mean():.2f}  per-slot argmax==slot {acc}')
    if h == 12: print(T)
