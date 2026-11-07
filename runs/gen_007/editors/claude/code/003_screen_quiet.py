# Hypothesis (H1): some neurons quantise to exactly 0 in llama.cpp's Q8_0 down projection on (almost) every token:
# |h_j| < absmax(32-block of h)/254. Reports per layer the quietest neurons by share of tokens with a nonzero Q8 value.
import numpy as np, json
from wsp.editor import *
res = {}
for L in range(12, 24):
    h = np.load(f"work/fwd_01/mlp{L}.npy").astype(np.float32)            # [T, 7680]
    T = h.shape[0]
    a = np.abs(h).reshape(T, 240, 32)
    amax = a.max(2, keepdims=True)
    nz = (a >= amax / 254.0).reshape(T, 7680)                          # quantised value != 0
    ratio = (a / np.maximum(amax, 1e-12)).reshape(T, 7680)
    share = nz.mean(0); mx = ratio.max(0)
    order = np.lexsort((mx, share))
    top = order[:8]
    res[L] = [(int(j), int(nz[:, j].sum()), round(float(mx[j]), 5)) for j in top]
    print(f"L{L}: neurons never nonzero: {int((share == 0).sum())}, nonzero on <=1 token: {int((nz.sum(0) <= 1).sum())}, "
          f"median share nonzero {np.median(share):.3f}; quietest (j, tokens nonzero/{T}, max ratio): {res[L][:5]}")
json.dump(res, open("work/quiet.json", "w"))
