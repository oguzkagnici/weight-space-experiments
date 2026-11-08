# Hypothesis (from ed_a gen 7): L23 neuron 881 is a near-duplicate of 331 and 5105, so its activation is a linear
# combination of theirs (h_881 ~ a*h_331 + b*h_5105) and its output can be folded into their w2 columns, freeing slot 881
# at a tiny output change. Fit a, b on pass-1 L23 inputs; report the residual relative to n881's own output.
import glob, numpy as np, torch
from wsp.editor import load
X = np.load(glob.glob("work/fwd_01/in_*layers.23*.npy")[0], mmap_mode="r")
x = torch.tensor(np.asarray(X[6::5], dtype=np.float32))
W1, W3 = load("model.layers.23.feed_forward.w1.weight").float(), load("model.layers.23.feed_forward.w3.weight").float()
silu = torch.nn.functional.silu
h = lambda j: silu(x @ W1[j]) * (x @ W3[j])
H = torch.stack([h(331), h(5105)], 1); y = h(881)
for j in (331, 5105): print(j, "cos w1", float(torch.cosine_similarity(W1[881], W1[j], 0)), "cos w3", float(torch.cosine_similarity(W3[881], W3[j], 0)))
coef = torch.linalg.lstsq(H, y[:, None]).solution.squeeze()
res = y - H @ coef
print("coef", coef.tolist(), "| rel residual", float(res.norm() / y.norm()), "| corr with 331", float(np.corrcoef(y, H[:, 0])[0, 1]))
np.save("work/fold_coef.npy", coef.numpy())
