# Hypothesis: (a) a larger set of constant-bias neurons and/or alpha>1 removes more of the early-letter offset with further gains; (b) attenuating layer-22 anti-correct heads (11,13,14,0,8) or boosting retrieval heads (12,15) adds gain. Lens/DLA counterfactual screening on DEV-512 (E1 validated: lens predicted logp +0.19, real +0.21 on unseen DEV).
import json, numpy as np
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows])
mask = np.arange(10)[None] < nopt[:, None]; m10 = nopt == 10
E = np.load('work/E_letters.npy'); w = load_original('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms24 = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T; Uc = U - U.mean(1, keepdims=True)
LL = np.load('work/lens_p1.npy')[24]
def ev(lg, idx=None):
    idx = np.arange(n) if idx is None else idx
    lg = np.where(mask, lg, -1e9)[idx]; a = ans[idx]
    lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
    return round((lg.argmax(1) == a).mean(), 4), round(float((lg[np.arange(len(a)), a] - lse).mean()), 3)
rec = json.load(open('work/bias_neurons.json'))
contrib = {}
for L in range(14, 24):
    js = [j for (l, j, *_) in rec if l == L]
    act = np.load(A + f'mlp{L}.npy').astype(np.float32)
    W2 = load_original(f'model.layers.{L}.feed_forward.w2.weight').numpy(); pw = load_original(f'model.layers.{L}.post_ffn_norm.weight').numpy()
    y = act @ W2.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
    G = (pw[:, None] * W2[:, js]).T @ Uc
    s = act[:, js] / (rq * rms24)[:, None]
    for i, j in enumerate(js): contrib[(L, j)] = s[:, i:i + 1] * G[i][None]
Lc = LL - LL.mean(1, keepdims=True); off = Lc[m10].mean(0); od = off / np.linalg.norm(off)
st = {k: ((c[m10] @ od).mean(), (c[m10] @ od).std(), c[np.arange(n), ans].mean()) for k, c in contrib.items()}
def selset(thr, ratio=1.0, hmax=0.03):
    return [k for k, (mu, sd, hp) in st.items() if mu > thr and sd < ratio * mu and abs(hp) < hmax]
E1 = [tuple(x[:2]) for x in json.load(open('work/sel_bias_neurons.json'))]
tE1 = sum(contrib[k] for k in E1)
print('base', ev(LL), 'E1', ev(LL - tE1))
lg = np.where(mask, LL - tE1, -1e9); r = lg - lg.mean(1, keepdims=True); print('remaining offset after E1', np.round((LL - tE1 - (LL - tE1).mean(1, keepdims=True))[m10].mean(0), 2))
for thr in [0.03, 0.04, 0.05, 0.08]:
    for ratio in [1.0, 2.0]:
        s = selset(thr, ratio); t = sum(contrib[k] for k in s)
        print(f'thr {thr} ratio {ratio} n {len(s)}', [ (a, ev(LL - a * t)) for a in [1.0, 1.3, 1.6]])
# heads layer 22
L = 22
x = np.load(A + f'in_model.layers.{L}.self_attn.o_proj.npy').astype(np.float32)
Wo = load_original(f'model.layers.{L}.self_attn.o_proj.weight').numpy(); pw = load_original(f'model.layers.{L}.post_operator_norm.weight').numpy()
y = x @ Wo.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
HD = {}
for h in range(16):
    yh = x[:, h * 128:(h + 1) * 128] @ Wo[:, h * 128:(h + 1) * 128].T
    HD[h] = (pw * yh / (rq * rms24)[:, None]) @ Uc
for name, hs, a in [('rm 11,13,14', [11, 13, 14], 1.0), ('rm 0,8,11,13,14', [0, 8, 11, 13, 14], 1.0), ('half 11,13,14', [11, 13, 14], 0.5), ('boost12 x0.2', [12], -0.2), ('boost12 x0.5', [12], -0.5), ('boost15 x0.5', [15], -0.5)]:
    t = sum(HD[h] for h in hs)
    print(name, 'alone', ev(LL - a * t), 'with E1', ev(LL - tE1 - a * t))
