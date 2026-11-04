# Hypothesis: the content-independent early-letter bias (A/B over-picked, H-J under-picked) is written by a few late components; direct logit attribution of each mixer/MLP output (layers 12-23) onto the option-letter unembeddings localizes it.
import json, numpy as np, torch
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows]); cor = np.array([r['correct'] for r in rows])
m10 = nopt == 10
E = np.load('work/E_letters.npy'); w = load('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T  # 2048x10
LL = np.load('work/lens_p1.npy')[24]
fin = LL[m10]; fin = fin - fin.mean(1, keepdims=True)
print('final mean centered letter logits (10-opt):', np.round(fin.mean(0), 2))
def dla(x):
    d = (x / rms[:, None]) @ U  # n x 10
    return d
comp = {}
for l in range(12, 24):
    comp[f'mix{l}'] = np.load(A + f'out_model.layers.{l}.post_operator_norm.npy')
    comp[f'mlp{l}'] = np.load(A + f'out_model.layers.{l}.post_ffn_norm.npy')
comp['h12'] = np.load(A + 'h12.npy')
tot = 0
print('component: mean centered letter contribution A..J (10-opt q) | bias score (A+B)-(I+J) | correct-letter help (d[ans]-mean)')
res = {}
for k, x in comp.items():
    d = dla(x); dc = d - d[:, :].mean(1, keepdims=True)
    mb = dc[m10].mean(0); tot = tot + mb
    help_ = (dc[np.arange(n), ans])[m10].mean()
    # content-dependent part: std across questions of centered
    res[k] = (mb, help_)
    print(f'{k:6s}', ' '.join(f'{v:+5.2f}' for v in mb), f'| {mb[0]+mb[1]-mb[8]-mb[9]:+5.2f} | {help_:+5.2f}  sd {dc[m10].std(0).mean():.2f}')
print('sum', np.round(tot, 2))
