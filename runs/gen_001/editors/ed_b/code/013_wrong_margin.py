# Hypothesis: on questions still wrong after E1, a specific late component (not the bias neurons) supplies most of the winning wrong letter's margin over the correct letter; decompose margin (pred - correct) by component.
import json, numpy as np
exec(open('code/011_screen.py').read().split('# heads layer 22')[0].split("for thr in")[0])
L2 = np.where(mask, LL - tE1, -1e9); pred2 = L2.argmax(1); wr = pred2 != ans
print('n wrong after E1', wr.sum())
comps = {'h12': np.load(A + 'h12.npy')}
for l in range(12, 24):
    comps[f'mix{l}'] = np.load(A + f'out_model.layers.{l}.post_operator_norm.npy'); comps[f'mlp{l}'] = np.load(A + f'out_model.layers.{l}.post_ffn_norm.npy')
idx = np.arange(n)
tot = 0; res = []
for k, x in comps.items():
    d = (x / rms24[:, None]) @ Uc
    if k.startswith('mlp'):
        L = int(k[3:]); d = d - sum(contrib[kk] for kk in E1 if kk[0] == L) if any(kk[0] == L for kk in E1) else d
    m = (d[idx, pred2] - d[idx, ans])[wr]
    res.append((m.mean(), k, (m > 0).mean())); tot += m.mean()
for r in sorted(res, reverse=True)[:10]: print(f'{r[1]:6s} mean margin contrib {r[0]:+.2f}  frac>0 {r[2]:.2f}')
for r in sorted(res)[:6]: print(f'{r[1]:6s} mean margin contrib {r[0]:+.2f}  frac>0 {r[2]:.2f}')
print('total', tot)
# head-level for L22
x = np.load(A + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)
Wo = load_original('model.layers.22.self_attn.o_proj.weight').numpy(); pw = load_original('model.layers.22.post_operator_norm.weight').numpy()
y = x @ Wo.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
for h in range(16):
    yh = x[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T; d = (pw * yh / (rq * rms24)[:, None]) @ Uc
    m = (d[idx, pred2] - d[idx, ans])[wr]
    if abs(m.mean()) > 0.1: print('L22 head', h, round(m.mean(), 2), 'frac>0', round((m > 0).mean(), 2))
# how often head12 alone picks correct among wrong-after-E1
y12 = x[:, 12*128:13*128] @ Wo[:, 12*128:13*128].T; d12 = np.where(mask, (pw * y12 / (rq * rms24)[:, None]) @ Uc, -1e9)
print('head12 alone correct on wrong-final:', (d12.argmax(1) == ans)[wr].mean(), ' on right-final:', (d12.argmax(1) == ans)[~wr].mean())
