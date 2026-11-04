# Hypothesis: head 12's letter readout has unequal per-letter gains (e.g. G amplified, A/B weak); equalizing them (lens counterfactual, multiplicative per-letter factor on head 12's push) would help beyond E1.
import json, numpy as np
exec(open('code/011_screen.py').read().split('# heads layer 22')[0].split("for thr in")[0])
x = np.load(A + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)
Wo = load_original('model.layers.22.self_attn.o_proj.weight').numpy(); pw = load_original('model.layers.22.post_operator_norm.weight').numpy()
y = x @ Wo.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
yh = x[:, 12*128:13*128] @ Wo[:, 12*128:13*128].T; d = (pw * yh / (rq * rms24)[:, None]) @ Uc
dm = np.where(mask, d, -1e9); pk = dm.argmax(1)
gain = np.array([d[pk == k, k].mean() for k in range(10)])
print('mean top push when head12 picks k:', np.round(gain, 2))
gold_gain = np.array([d[ans == k, k].mean() - np.delete(d[ans == k], k, 1).mean() for k in range(10)])
print('mean push on gold k minus others, by gold letter:', np.round(gold_gain, 2))
rng = np.random.default_rng(3); L2 = LL - tE1
for rep in range(3):
    p = rng.permutation(n); h1, h2 = p[:n//2], p[n//2:]
    g = np.array([d[h1][ans[h1] == k, k].mean() - np.delete(d[h1][ans[h1] == k], k, 1).mean() for k in range(10)])
    f = g.mean() / g
    for a in [0.5, 1.0]:
        fac = 1 + a * (f - 1)
        print(rep, a, 'E1', ev(L2, h2), 'E1+gain-eq', ev(L2 - d + d * fac[None], h2), np.round(fac, 2))
