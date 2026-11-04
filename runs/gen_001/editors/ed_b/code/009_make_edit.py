# Hypothesis: projecting the write vectors (w2 columns) of the selected constant early-letter-bias neurons off the (centered) option-letter readout subspace removes their letter offset while keeping their other outputs; usage: py code/009_make_edit.py <sel.json> <alpha>
import sys, json, numpy as np, torch
from wsp.editor import *
sel = json.load(open(sys.argv[1])); alpha = float(sys.argv[2])
E = torch.tensor(np.load('work/E_letters.npy'))  # 10x2048
w = load_original('model.embedding_norm.weight')
for name in edited(): drop_edit(name)
by = {}
for l, j, *_ in sel: by.setdefault(int(l), []).append(int(j))
for L, js in sorted(by.items()):
    W2 = load_original(f'model.layers.{L}.feed_forward.w2.weight')
    pw = load_original(f'model.layers.{L}.post_ffn_norm.weight')
    M = (E * w[None] * pw[None]).T  # 2048x10: letter readout through post-norm + final-norm gains
    Mc = M - M.mean(1, keepdim=True)
    P = Mc @ torch.linalg.pinv(Mc)  # projector onto centered letter subspace (rank 9)
    before = (Mc.T @ W2[:, js])
    W2n = W2.clone()
    W2n[:, js] = W2[:, js] - alpha * (P @ W2[:, js])
    after = (Mc.T @ W2n[:, js])
    print(L, js, 'letter push norm before %.4f after %.4f' % (before.norm(), after.norm()), 'col rel change', [round(float((W2n[:, j] - W2[:, j]).norm() / W2[:, j].norm()), 3) for j in js])
    print(save_edit(f'model.layers.{L}.feed_forward.w2.weight', W2n))
print(edited())
