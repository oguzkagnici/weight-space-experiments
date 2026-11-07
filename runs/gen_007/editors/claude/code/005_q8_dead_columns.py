# Hypothesis (H3): some w2 columns quantise to all (or almost all) zeros in Q8_0 (|w| < row-block absmax/254 in every row),
# so the neuron already contributes (almost) nothing in the evaluated model, whatever its activation. Weights only (bf16-rounded).
import numpy as np, torch
from wsp.editor import *
for L in range(0, 24):
    w2 = load_original(f"model.layers.{L}.feed_forward.w2.weight").to(torch.bfloat16).float().numpy()   # [2048, 7680]
    a = np.abs(w2).reshape(2048, 240, 32)
    amax = a.max(2, keepdims=True)
    nz = (a >= amax / 254.0).reshape(2048, 7680)
    cnt = nz.sum(0)
    order = np.argsort(cnt)
    print(f"L{L:2d}: fewest nonzero Q8 entries per column (of 2048): {[(int(j), int(cnt[j])) for j in order[:4]]}, median {int(np.median(cnt))}")
