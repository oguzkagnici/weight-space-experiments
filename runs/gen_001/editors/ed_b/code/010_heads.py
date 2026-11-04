# Hypothesis: the correct-letter signal of the late attention layers comes from a few heads (per-head DLA via o_proj inputs); some heads may instead carry letter bias or push wrong letters.
import json, numpy as np
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows])
m10 = nopt == 10
E = np.load('work/E_letters.npy'); w = load('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms24 = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T; Uc = U - U.mean(1, keepdims=True)
LL = np.load('work/lens_p1.npy')[24]; Lc = LL - LL.mean(1, keepdims=True); off = Lc[m10].mean(0); od = off / np.linalg.norm(off)
for L in [12, 14, 16, 18, 20, 22]:
    x = np.load(A + f'in_model.layers.{L}.self_attn.o_proj.npy').astype(np.float32)
    Wo = load(f'model.layers.{L}.self_attn.o_proj.weight').numpy(); pw = load(f'model.layers.{L}.post_operator_norm.weight').numpy()
    y = x @ Wo.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
    out = np.load(A + f'out_model.layers.{L}.post_operator_norm.npy')
    err = np.linalg.norm(out - pw * y / rq[:, None]) / np.linalg.norm(out)
    line = []
    for h in range(16):
        yh = x[:, h * 128:(h + 1) * 128] @ Wo[:, h * 128:(h + 1) * 128].T
        d = (pw * yh / (rq * rms24)[:, None]) @ Uc
        hp = d[np.arange(n), ans].mean(); bias = (d[m10] @ od).mean(); sd = d.std(1).mean()
        acc = (np.where(np.arange(10)[None] < nopt[:, None], d, -1e9).argmax(1) == ans).mean()
        line.append(f'h{h}:{hp:+.2f}/{bias:+.2f}/{acc:.2f}')
    print(f'L{L} (recon err {err:.3f}) help/bias/acc:', ' '.join(line))
