# Hypothesis: MLP22 (DLA: correct-letter help -1.19, early-letter bias +2.46) suppresses the letter that attention 22 retrieves; a small set of its neurons does this and can be ablated.
import json, numpy as np, torch
from wsp.editor import *
A = 'work/fwd_01/score_1_acts/original/'
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows]); cor = np.array([r['correct'] for r in rows])
m10 = nopt == 10
E = np.load('work/E_letters.npy'); w = load('model.embedding_norm.weight').numpy()
h24 = np.load(A + 'h24.npy'); rms24 = np.sqrt((h24 ** 2).mean(1) + 1e-5)
U = (E * w[None]).T
Uc = U - U.mean(1, keepdims=True)
def dc(x):
    d = (x / rms24[:, None]) @ U; return d - d.mean(1, keepdims=True)
a22 = dc(np.load(A + 'out_model.layers.22.post_operator_norm.npy'))
m22 = dc(np.load(A + 'out_model.layers.22.post_ffn_norm.npy'))
# per question correlation across options between attn22 and mlp22 letter pushes
cc = []
for i in range(n):
    k = nopt[i]; cc.append(np.corrcoef(a22[i, :k], m22[i, :k])[0, 1])
cc = np.array(cc)
print('per-q corr(attn22 push, mlp22 push) across letters: mean %.3f median %.3f' % (np.nanmean(cc), np.nanmedian(cc)))
top = a22.argmax(1)
print('mlp22 push on attn22-top letter (mean centered): %.2f ; on correct: %.2f ; attn22 top==ans frac %.3f' % (m22[np.arange(n), top].mean(), m22[np.arange(n), ans].mean(), (top == ans).mean()))
# per-neuron decomposition
L = 22
act = np.load(A + f'mlp{L}.npy').astype(np.float32)  # n x 7680
W2 = load(f'model.layers.{L}.feed_forward.w2.weight').numpy()  # 2048x7680
pw = load(f'model.layers.{L}.post_ffn_norm.weight').numpy()
y = act @ W2.T
rq = np.sqrt((y ** 2).mean(1) + 1e-5)
out = np.load(A + f'out_model.layers.{L}.post_ffn_norm.npy')
print('reconstruct check rel err', np.linalg.norm(out - pw * y / rq[:, None]) / np.linalg.norm(out))
G = (pw[:, None] * W2).T @ Uc  # 7680 x 10 : letter push per unit act
scale = 1 / (rq * rms24)  # per q
contrib = act[:, :, None] * G[None] * scale[:, None, None]  # n x 7680 x 10  (~ 512*7680*10*4 = 157MB)
help_n = contrib[np.arange(n), :, ans].mean(0)  # mean push on correct letter per neuron
bias_n = contrib[m10].mean(0)  # 7680x10
eb = bias_n[:, :2].sum(1) - bias_n[:, 8:].sum(1)
# push on attn22-top letter
toppush = contrib[np.arange(n), :, top].mean(0)
o = np.argsort(help_n)
print('total help', help_n.sum(), 'total top push', toppush.sum())
print('most correct-suppressing neurons: id help toppush earlybias meanact')
for j in o[:25]:
    print(j, f'{help_n[j]:+.3f} {toppush[j]:+.3f} {eb[j]:+.3f} {act[:, j].mean():+.3f}')
print('cumulative help of top-k suppressors:', [(k, round(help_n[o[:k]].sum(), 3), round(eb[o[:k]].sum(), 3)) for k in [5, 10, 20, 50, 100, 200]])
oe = np.argsort(-eb)
print('top early-bias neurons:', [(int(j), round(eb[j], 3), round(help_n[j], 3)) for j in oe[:15]])
np.save('work/mlp22_help.npy', help_n); np.save('work/mlp22_eb.npy', eb); np.save('work/mlp22_toppush.npy', toppush)
