# Pick the slot for Unit A: the L23 neurons whose current output is smallest (mean |act| x |w2 col| over 8,000 recorded
# tokens), excluding the parent's units (1228, 3401, 3793, 6986) and ed_a's free duplicates (331, 881, 5105).
# Also report each candidate's Q8 block headroom (mean block absmax of its w2 column rows).
import numpy as np, torch
from wsp.editor import *
X = torch.tensor(np.load("work/fwd_02/in_model.layers.23.feed_forward.npy").astype(np.float32))
idx = torch.tensor(np.random.default_rng(0).choice(len(X), 8000, replace=False)); Xs = X[idx]
W1 = load_original("model.layers.23.feed_forward.w1.weight"); W3 = load_original("model.layers.23.feed_forward.w3.weight")
W2 = load_original("model.layers.23.feed_forward.w2.weight")
Z = Xs @ W1.T; A = Z * torch.sigmoid(Z) * (Xs @ W3.T)
q = (A.abs().mean(0) * W2.norm(dim=0)).numpy(); mx = (A.abs().max(0).values * W2.norm(dim=0)).numpy()
bad = {1228, 3401, 3793, 6986, 331, 881, 5105}
order = [j for j in np.argsort(q) if j not in bad][:8]
for j in order:
    b0 = (j // 32) * 32; blk = W2[:, b0:b0 + 32].clone(); blk[:, j - b0] = 0
    print(f"n{j}: mean|out| {q[j]:.5f}, max|out| {mx[j]:.3f}, block absmax mean {float(blk.abs().max(1).values.mean()):.4f}")
print(f"median neuron mean|out| {np.median(q):.4f}")
