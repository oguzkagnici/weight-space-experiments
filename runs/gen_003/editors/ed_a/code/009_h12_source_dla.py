# Hypothesis: L22 heads' attention to the "(X" letter tokens is content-free (acc ~chance) but writes a strong,
# position-biased letter vote (values at "(X" carry the letter identity directly) -> a source of the early-letter prior.
# Decompose each L22 head's output at the answer position by source token class and read the letter logits (DLA).
import json, numpy as np, torch
from wsp.editor import *
D1, D2 = 'work/fwd_01/', 'work/fwd_02/'
ids = json.load(open('work/pass2_ids.json')); SP = json.load(open('work/spans_p1.json'))
N = len(ids)
off1 = np.load(D1 + 'offsets_tok.npy'); off2 = np.load(D2 + 'offsets.npy')
gold = np.load('work/p1_gold.npy')[:N]; nopt = np.load('work/p1_nopt.npy')[:N]; LP = np.load('work/p1_letterlogp.npy')[:N]
mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1)
A = np.load(D1 + 'attn_last_L22.npy').astype(np.float32)
V = np.load(D2 + 'out_model.layers.22.self_attn.v_proj.npy', mmap_mode='r')
oin = np.load(D1 + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)[:N]
h24 = np.load(D1 + 'h24.npy').astype(np.float32)[:N]
WO = load_base('model.layers.22.self_attn.o_proj.weight').numpy()
gpost = load_base('model.layers.22.post_operator_norm.weight').numpy()
wn = load_base('model.embedding_norm.weight').numpy()
EL = load_base('model.embed_tokens.weight')[LETTER_IDS].numpy()
R = (EL * wn) * gpost  # 10 x 2048 readout of a pre-post-norm attention write (before dividing by rms's)
classes = ['letter', 'text', 'nl', 'other']
dla = np.zeros((N, 16, 4, 10)); gate = np.zeros((N, 16))
for i, q in enumerate(ids):
    a = A[off1[i]:off1[i + 1]]  # T x 16
    v = V[off2[i]:off2[i + 1]].astype(np.float32).reshape(-1, 4, 128)
    T = a.shape[0]
    cls = np.full(T, 3)
    for k, (t0, t1) in enumerate(SP[q]['spans']):
        cls[t0:t0 + 2] = 0; cls[t0 + 2:t1 + 1] = 1; cls[t1 + 1] = 2 if k < len(SP[q]['spans']) - 1 else cls[t1 + 1]
    total = WO @ oin[i]
    s_attn = np.sqrt((total ** 2).mean() + 1e-5); s_fin = np.sqrt((h24[i] ** 2).mean() + 1e-5)
    for h in range(16):
        o_full = (a[:, h:h + 1] * v[:, h // 4]).sum(0)
        g = oin[i, h * 128:(h + 1) * 128] @ o_full / (o_full @ o_full + 1e-9)
        gate[i, h] = g
        for c in range(4):
            sel = cls == c
            o_c = g * (a[sel, h:h + 1] * v[sel, h // 4]).sum(0)
            w = WO[:, h * 128:(h + 1) * 128] @ o_c
            dla[i, h, c] = R @ w / s_attn / s_fin
np.save('work/p2_l22_dla_by_class.npy', dla.astype(np.float32))
print('gate mean by head', np.round(gate.mean(0), 2))
dc = dla - np.where(mask[:, None, None], dla, 0).sum(-1, keepdims=True) / nopt[:, None, None, None]
m10 = nopt == 10
print('10-option questions:', m10.sum())
for h in range(16):
    row = []
    for c in range(4):
        x = dc[:, h, c]
        g_ = x[np.arange(N), gold].mean()
        bias = x[m10].mean(0)
        pk = np.where(mask, x, -1e9).argmax(1)
        row.append(f'{classes[c]}: gold {g_:+.2f} acc {(pk==gold).mean():.2f} bias[{" ".join(f"{b:+.1f}" for b in bias)}]')
    tot = np.abs(dc[:, h]).sum(-1).mean()
    if tot > 0.5 * 4: print(f'h{h:2d} ' + '\n     '.join(row))
