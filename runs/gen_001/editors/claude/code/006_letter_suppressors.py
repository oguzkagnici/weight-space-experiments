# hypothesis H2: L22/L23 MLPs contain letter-specific neurons that suppress the CURRENT LEADING letter (content-
# dependent negative feedback), and their gains differ across letters (late letters suppressed harder) -> net bias.
# Test: for each letter-specific neuron, activation when its letter leads the stream entering the MLP vs not;
# then the per-letter total suppression gain.
import json, numpy as np, torch
from wsp.editor import *

Q = questions()
def load_pass(k):
    base = f"work/fwd_{k:02d}/score_{k}_acts/original"
    idx = json.load(open(f"{base}/index.json"))
    return (lambda key: np.load(f"{base}/{idx['files'][key][0]}").astype(np.float64)), \
           [json.loads(l) for l in open(f"work/fwd_{k:02d}/score_{k}_original.jsonl")]
A1, r1 = load_pass(1); A2, r2 = load_pass(2)
assert [r["id"] for r in r1] == [r["id"] for r in r2] and [r["probs"] for r in r1] == [r["probs"] for r in r2]
n = len(r2); nopt = np.array([len(Q[r["id"]]["options"]) for r in r2]); ans = np.array([r["answer"] for r in r2])
pred = np.array([r["pred"] for r in r2])
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double()
wf = load_base("model.embedding_norm.weight").double()
Ec = E - E.mean(0, keepdim=True)

def lens_top(h):
    h = torch.tensor(h); hn = h / torch.sqrt((h * h).mean(-1, keepdim=True) + 1e-5) * wf
    L = (hn @ E.T).numpy()
    for i in range(n): L[i, nopt[i]:] = -1e9
    return L.argmax(1), L

# stream entering the L22 MLP = h22 + post-normed attention output of L22 (from pass 1's o_proj input)
z = torch.tensor(A1("in:model.layers.22.self_attn.o_proj"))
Wo = load_base("model.layers.22.self_attn.o_proj.weight").double()
wpo = load_base("model.layers.22.post_operator_norm.weight").double()
mix = z @ Wo.T; attn22 = (mix / torch.sqrt((mix * mix).mean(-1, keepdim=True) + 1e-5) * wpo).numpy()
pre = {22: A2("h22") + attn22, 23: None}
top_pre22, _ = lens_top(pre[22])
top_h23, _ = lens_top(A2("h23"))
print(f"lens top letter entering L22 MLP = correct {np.mean(top_pre22 == ans):.3f}; = final pred {np.mean(top_pre22 == pred):.3f}")

rep = []
for l, top in ((22, top_pre22), (23, top_h23)):
    act = torch.tensor(A2(f"mlp{l}"))
    W2 = load_base(f"model.layers.{l}.feed_forward.w2.weight").double()
    wpf = load_base(f"model.layers.{l}.post_ffn_norm.weight").double()
    ro = (((W2 * wpf[:, None]) * wf[:, None]).T @ Ec.T).numpy()           # 7680 x 10
    mo = act @ W2.T; rmso = torch.sqrt((mo * mo).mean(-1, keepdim=True) + 1e-5)
    h24 = torch.tensor(A2("h24")); s = (1 / rmso / torch.sqrt((h24 ** 2).mean(-1, keepdim=True) + 1e-5))[:, 0].numpy()
    a = act.numpy()
    rc = ro - ro.mean(1, keepdims=True)
    spec = np.abs(rc).max(1) / (np.abs(rc).sum(1) + 1e-12)                 # share of the dominant letter
    let = np.abs(rc).argmax(1)
    for j in np.where(spec > 0.3)[0]:
        L = let[j]
        push = a[:, j] * rc[j, L] * s                                       # push on its own letter per question
        if np.abs(push).mean() < 0.02: continue
        lead = top == L
        if lead.sum() < 3 or (~lead).sum() < 3: continue
        rep.append((l, j, "ABCDEFGHIJ"[L], push.mean(), push[lead].mean(), push[~lead].mean(), lead.sum(),
                    push[ans == L].mean() if (ans == L).sum() else np.nan))
rep.sort(key=lambda r: r[3])
print("\nletter-specific neurons (|mean push| >= .02): layer neuron letter | mean push | when letter LEADS | when not | "
      "#leads | when letter is CORRECT")
for r in rep[:30]:
    print(f"  L{r[0]} n{r[1]:5d} {r[2]} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.3f} | {r[6]:3d} | {r[7]:+.3f}")
print("  ... boosters:")
for r in rep[-10:]:
    print(f"  L{r[0]} n{r[1]:5d} {r[2]} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.3f} | {r[6]:3d} | {r[7]:+.3f}")
print("\nper-letter total of letter-specific pushes (L22+L23): mean push / when leading")
for Lt in "ABCDEFGHIJ":
    rr = [r for r in rep if r[2] == Lt]
    print(f"  {Lt}: {len(rr):2d} neurons  mean {sum(r[3] for r in rr):+.2f}  when-leading {sum(r[4] for r in rr):+.2f}  "
          f"when-not {sum(r[5] for r in rr):+.2f}")
json.dump([[int(r[0]), int(r[1]), r[2], float(r[3]), float(r[4]), float(r[5])] for r in rep], open("work/letter_neurons.json", "w"))
