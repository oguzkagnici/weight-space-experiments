# Hypothesis: (tool) build an edit that removes a fraction f of the centered direct letter readout (9-dim span of
# wn*E[A..J] minus mean, taken in post_ffn_norm-scaled space) from chosen MLP w2 columns. Usage:
#   py code/022_make_edit.py <spec.json>   spec = {"cols": [[layer, neuron, f], ...]}  (drops all previous edits first)
import json, sys, numpy as np, torch
from wsp.editor import *
spec = json.load(open(sys.argv[1]))
for n in edited(): drop_edit(n)
wn = load_base('model.embedding_norm.weight')
EL = load_base('model.embed_tokens.weight')[LETTER_IDS] * wn  # 10 x 2048
Mc = EL - EL.mean(0, keepdim=True)
U, S, Vt = torch.linalg.svd(Mc, full_matrices=False)
Qb = Vt[:9]  # 9 x 2048 orthonormal basis of centered letter readout
by_layer = {}
for l, n, f in spec['cols']: by_layer.setdefault(l, []).append((n, f))
for l, lst in sorted(by_layer.items()):
    name = f'model.layers.{l}.feed_forward.w2.weight'
    W = load_original(name).clone()
    g = load_base(f'model.layers.{l}.post_ffn_norm.weight')
    for n, f in lst:
        u = g * W[:, n]
        r = Qb.T @ (Qb @ u)
        W[:, n] = (u - f * r) / g
        print(f'  L{l} n{n} f={f}: removed readout norm {float(r.norm()):.3f} of column norm {float(u.norm()):.3f}')
    print(save_edit(name, W))
print('edited:', edited())
