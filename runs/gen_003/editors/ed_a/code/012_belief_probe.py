# Hypothesis: the residual at option-end newline tokens (h14, h21) holds a correctness belief along a mean-difference
# direction that picks gold better than L22.H12's attention does (unused belief) -> room for a key-side edit.
import json, numpy as np
from wsp.editor import *
D2 = 'work/fwd_02/'
ids = json.load(open('work/pass2_ids.json')); SP = json.load(open('work/spans_p1.json'))
N = len(ids); off2 = np.load(D2 + 'offsets.npy')
gold = np.load('work/p1_gold.npy')[:N]; nopt = np.load('work/p1_nopt.npy')[:N]; LP = np.load('work/p1_letterlogp.npy')[:N]
mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1)
H12m = np.load('work/p1_spanmass_L22.npy')[:N, :, 12]  # span (no newline) mass; recompute with newline below
K = np.load(D2 + 'out_model.layers.22.self_attn.k_norm.npy', mmap_mode='r')
print('k_norm shape', K.shape)
res = {}
for layer in [14, 21]:
    H = np.load(D2 + f'h{layer}.npy', mmap_mode='r')
    X = np.zeros((N, 10, 2048), np.float32); Kn = np.zeros((N, 10, 128), np.float32)
    for i, q in enumerate(ids):
        for k, (t0, t1) in enumerate(SP[q]['spans']):
            p = off2[i] + t1 + 1 if k < nopt[i] - 1 else off2[i] + t1  # newline token (last option: its last text token)
            X[i, k] = H[p]
            if layer == 21: Kn[i, k] = np.asarray(K[p]).reshape(4, 128)[3] if K.ndim == 2 else np.asarray(K[p])[3]
    # center within question
    Xc = X - (X * mask[..., None]).sum(1, keepdims=True) / nopt[:, None, None]
    rng = np.random.RandomState(1)
    accs = []
    for rep in range(10):
        f = rng.rand(N) < .5
        for tr, te in [(f, ~f), (~f, f)]:
            G = Xc[tr][np.arange(tr.sum()), gold[tr]]
            Oth = (Xc[tr] * mask[tr][..., None]).sum(1) - G
            d = G.mean(0) - (Oth / (nopt[tr] - 1)[:, None]).mean(0)
            s = np.where(mask[te], Xc[te] @ d, -1e9)
            accs.append((s.argmax(1) == gold[te]).mean())
    print(f'h{layer} newline mean-diff probe: CV acc {np.mean(accs):.3f} (+-{np.std(accs)/np.sqrt(len(accs)):.3f}); final acc {(final==gold).mean():.3f}')
    np.save(f'work/p2_nlX_h{layer}.npy', X)
np.save('work/p2_nlK22g3.npy', Kn)
