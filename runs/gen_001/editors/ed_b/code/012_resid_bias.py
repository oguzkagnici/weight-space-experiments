# Hypothesis: after E1 some letter offset remains; estimate how much accuracy a full removal of the remaining constant offset would still buy (half-split), to decide whether to hunt more bias sources.
import json, numpy as np
exec(open('code/011_screen.py').read().split('# heads layer 22')[0].split("for thr in")[0])
rng = np.random.default_rng(5)
L2 = LL - tE1
for rep in range(4):
    p = rng.permutation(n); h1, h2 = p[:n//2], p[n//2:]
    f1 = h1[m10[h1]]; o = (L2 - L2.mean(1, keepdims=True))[f1].mean(0)
    o0 = (LL - LL.mean(1, keepdims=True))[f1].mean(0)
    print(rep, 'test: base', ev(LL, h2), 'E1', ev(L2, h2), 'E1+full debias', ev(L2 - o, h2), 'base+full debias', ev(LL - o0, h2))
