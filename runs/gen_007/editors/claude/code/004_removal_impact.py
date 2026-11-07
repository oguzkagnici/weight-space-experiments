# Hypothesis (H2): how early a repurposed neuron makes traces diverge scales with its original contribution relative to
# the MLP output. Score s_j = mean over recorded tokens of |h_j| * ||g_post * w2_j|| / sqrt(sum_i h_i^2 * ||g_post * w2_i||^2)
# (incoherent-sum proxy of the output norm; arithmetic on recorded activations and weights only).
# Calibration: gen-6 neurons L21 n6823 (median divergence 7% of trace), L23 n74 (22%), L23 n4526+n7602+n810 (14%).
import numpy as np, json
from wsp.editor import *
out = {}
for L in range(12, 24):
    h = np.load(f"work/fwd_01/mlp{L}.npy").astype(np.float32)
    w2 = load_original(f"model.layers.{L}.feed_forward.w2.weight").numpy()       # [2048, 7680]
    g = load_original(f"model.layers.{L}.post_ffn_norm.weight").numpy()
    cn = np.linalg.norm(w2 * g[:, None], axis=0)                                 # [7680]
    c = np.abs(h) * cn[None, :]
    tot = np.sqrt((c ** 2).sum(1, keepdims=True))
    s = (c / tot).mean(0)
    out[L] = s
    order = np.argsort(s)
    print(f"L{L}: median s {np.median(s):.5f}; lowest {[(int(j), round(float(s[j]), 6)) for j in order[:4]]}")
np.save("work/impact.npy", np.stack([out[L] for L in range(12, 24)]))
print("calibration: L21 n6823", round(float(out[21][6823]), 6), "| L23 n74", round(float(out[23][74]), 6),
      "| L23 n4526+n7602+n810", round(float(out[23][[4526, 7602, 810]].sum()), 6), "| L20 n5414 (parent's edit)", round(float(out[20][5414]), 6))
