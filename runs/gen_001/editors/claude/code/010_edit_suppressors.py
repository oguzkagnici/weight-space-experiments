# hypothesis H2 (edit): equalise the letter-specific "leader suppression" by scaling the DIRECT LETTER READOUT of the
# late-letter suppressor neurons by alpha (only the component of their w2 column that the tied letter rows read through
# the post-FFN norm weight and final norm weight; everything else the neuron writes is unchanged).
# usage: py code/010_edit_suppressors.py ALPHA [L23ALPHA]
import sys, json, torch
from wsp.editor import *

alpha = float(sys.argv[1]); alpha23 = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "none" else None
beta = float(sys.argv[3]) if len(sys.argv) > 3 else None   # letter-readout scale of the early-letter boosters
BOOSTERS = {22: {"A": 2212}, 23: {"A": 1685, "A2": 2595, "B": 1439}}
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double()
wf = load_base("model.embedding_norm.weight").double()
Ec = E - E.mean(0, keepdim=True)
NEURONS = {22: {"C": 1812, "D": 464, "E": 7357, "F": 6492, "G": 3344, "H": 397, "I": 2786, "J": 3310},
           23: {"C": 1192, "D": 1876, "E": 2758, "F": 3520, "G": 1107, "H": 358, "I": 6512, "J": 1041}}
for name in edited():
    drop_edit(name)
for l in (22, 23):
    a = alpha if l == 22 else alpha23
    neurons = dict(NEURONS[l]) if a is not None else {}
    scale = {j: a for j in neurons.values()}
    if beta is not None:
        for nm, j in BOOSTERS[l].items():
            neurons["boost_" + nm] = j; scale[j] = beta
    if not neurons:
        continue
    key = f"model.layers.{l}.feed_forward.w2.weight"
    W2 = load_original(key).double()
    wpf = load_original(f"model.layers.{l}.post_ffn_norm.weight").double()
    M = Ec * (wf * wpf)[None, :]                       # 10 x 2048: letter readout of a column (per unit activation)
    P = torch.linalg.pinv(M) @ M                       # projector onto the readout row space (rank 9)
    for letter, j in neurons.items():
        c = W2[:, j].clone()
        before = (M @ c); before = before - before.mean()
        W2[:, j] = c - (1 - scale[j]) * (P @ c)
        after = (M @ W2[:, j]); after = after - after.mean()
        print(f"L{l} n{j} ({letter}): letter readout {float(before.abs().max()):.3f} -> {float(after.abs().max()):.3f}, "
              f"column change {float((W2[:, j] - c).norm() / c.norm()):.1%}")
    print(save_edit(key, W2.float()))
print("edited:", edited())
