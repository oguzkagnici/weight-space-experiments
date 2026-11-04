# hypothesis: some attention heads at the answer position write a letter-POSITION preference (early letters up,
# late letters down) independent of content, and/or push wrong options; find them by direct logit attribution (DLA)
# of each head's recorded output through o_proj -> post_operator_norm -> (final norm scale) -> tied letter rows.
import json, sys, numpy as np, torch
from wsp.editor import *

k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
base = f"work/fwd_{k:02d}/score_{k:02d}_acts/original"
idx = json.load(open(f"{base}/index.json"))
rows = [json.loads(l) for l in open(f"work/fwd_{k:02d}/score_{k:02d}_original.jsonl")]
Q = questions()
n = len(rows)
nopt = np.array([len(Q[r["id"]]["options"]) for r in rows])
ans = np.array([r["answer"] for r in rows]); pred = np.array([r["pred"] for r in rows])
correct = np.array([r["correct"] for r in rows])
print(f"n={n} acc={correct.mean():.3f}  10-option share={np.mean(nopt == 10):.2f}")

E = load_base("model.embed_tokens.weight")[LETTER_IDS].double()           # 10 x 2048
wf = load_base("model.embedding_norm.weight").double()
def A(key):
    return np.load(f"{base}/{idx['files'][key][0]}").astype(np.float64)

def option_logits(h):            # h: n x 2048 residual -> n x 10 letter logits (final norm applied)
    h = torch.tensor(h)
    hn = h / torch.sqrt((h * h).mean(-1, keepdim=True) + 1e-5) * wf
    return (hn @ E.T).numpy()

def masked_acc(L):
    L = L.copy()
    for i in range(n): L[i, nopt[i]:] = -1e9
    return (L.argmax(1) == ans).mean(), L

print("\nlogit lens (accuracy over option letters, mean rank of correct):")
lens = {}
for l in [8, 12, 16, 18, 20, 22, 24]:
    a, L = masked_acc(option_logits(A(f"h{l}")))
    rank = np.array([(L[i, :nopt[i]] > L[i, ans[i]]).sum() for i in range(n)])
    dist = np.bincount(L.argmax(1), minlength=10)
    lens[l] = L
    print(f"  after L{l:2d}: acc {a:.3f}  mean rank {rank.mean():.2f}  pred letters {dist.tolist()}")

h24 = A("h24"); rms24 = np.sqrt((h24 ** 2).mean(-1, keepdims=True) + 1e-5)
print("\nper-head DLA at the answer position (letter logit units, centred over the question's options):")
res = []
for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    z = torch.tensor(A(f"in:model.layers.{l}.self_attn.o_proj"))            # n x 2048 (16 heads x 128)
    Wo = load_base(f"model.layers.{l}.self_attn.o_proj.weight").double()      # 2048 x 2048
    wpost = load_base(f"model.layers.{l}.post_operator_norm.weight").double()
    mix = z @ Wo.T
    rms_mix = torch.sqrt((mix * mix).mean(-1, keepdim=True) + 1e-5)
    for h in range(16):
        c = (z[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T) / rms_mix * wpost   # n x 2048 contribution
        d = ((c * wf / torch.tensor(rms24)) @ E.T).numpy()                          # n x 10
        dc = np.zeros(n); pos = np.zeros((n, 10)); cnt = np.zeros(10)
        for i in range(n):
            m = nopt[i]; v = d[i, :m] - d[i, :m].mean()
            dc[i] = v[ans[i]]                       # push toward the correct letter (centred)
            pos[i, :m] = v; cnt[:m] += 1
        posm = pos.sum(0) / np.maximum(cnt, 1)
        slope = np.polyfit(np.arange(10), posm, 1)[0]
        res.append((l, h, dc.mean(), dc[correct == 0].mean(), np.abs(d - d.mean(1, keepdims=True)).mean(), slope, posm))
res.sort(key=lambda r: -r[4])
print("  layer head | to-correct (all / wrong qs) | mean |letter DLA| | position slope per letter | A..J profile")
for r in res[:25]:
    print(f"  L{r[0]:2d} H{r[1]:2d} | {r[2]:+.3f} / {r[3]:+.3f} | {r[4]:.3f} | {r[5]:+.4f} | " + " ".join(f"{x:+.2f}" for x in r[6]))
np.save("work/head_dla_summary.npy", np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in res]))
print("\nmost position-biased heads (slope):")
for r in sorted(res, key=lambda r: r[5])[:8]:
    print(f"  L{r[0]:2d} H{r[1]:2d} slope {r[5]:+.4f}  to-correct {r[2]:+.3f}  |DLA| {r[4]:.3f}")
