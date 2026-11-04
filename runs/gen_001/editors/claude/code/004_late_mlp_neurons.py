# hypothesis H1: a few late MLP neurons (layers 21-23) write a content-independent letter prior (A/B up, I/J down) at
# the answer position: they fire on (almost) every question and their output direction, read through the tied letter
# rows, has an early-vs-late letter profile. Rank neurons by (mean activation x letter-profile of their w2 column).
import json, numpy as np, torch
from wsp.editor import *

k = 2
base = f"work/fwd_{k:02d}/score_{k}_acts/original"
idx = json.load(open(f"{base}/index.json"))
rows = [json.loads(l) for l in open(f"work/fwd_{k:02d}/score_{k}_original.jsonl")]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r["id"]]["options"]) for r in rows]); m10 = nopt == 10
ans = np.array([r["answer"] for r in rows]); pred = np.array([r["pred"] for r in rows]); cor = pred == ans
A = lambda key: np.load(f"{base}/{idx['files'][key][0]}").astype(np.float64)
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double()
wf = load_base("model.embedding_norm.weight").double()
h = {l: torch.tensor(A(f"h{l}")) for l in (21, 22, 23, 24)}
rms24 = torch.sqrt((h[24] ** 2).mean(-1, keepdim=True) + 1e-5)
Ec = E - E.mean(0, keepdim=True)          # centred letter readout (10-option questions)
prof_dir = torch.linspace(-1, 1, 10).double()  # early (+) .. late (-) is -prof_dir

def lp(c):                                 # mean centred letter logit profile (10-option qs)
    d = ((c[m10] * wf / rms24[m10]) @ Ec.T).numpy()
    return d.mean(0)
print("profiles (10-option qs) A..J")
print("  L21->L22 block (h22-h21)  ", " ".join(f"{x:+.2f}" for x in lp(h[22] - h[21])))
print("  L22 block (h23-h22)       ", " ".join(f"{x:+.2f}" for x in lp(h[23] - h[22])))
print("  L23 block (h24-h23)       ", " ".join(f"{x:+.2f}" for x in lp(h[24] - h[23])))

out = {}
for l in (21, 22, 23):
    act = torch.tensor(A(f"mlp{l}"))                                  # n x 7680 (w2 input)
    W2 = load_base(f"model.layers.{l}.feed_forward.w2.weight").double()   # 2048 x 7680
    wpf = load_base(f"model.layers.{l}.post_ffn_norm.weight").double()
    mo = act @ W2.T; rmso = torch.sqrt((mo * mo).mean(-1, keepdim=True) + 1e-5)
    tot = mo / rmso * wpf
    print(f"  L{l} MLP total            ", " ".join(f"{x:+.2f}" for x in lp(tot)))
    # per-neuron letter readout of its column (after post-ffn norm weight and final norm weight), centred
    col = ((W2 * wpf[:, None]) * wf[:, None]).T @ Ec.T                # 7680 x 10 (per unit activation)
    scale = (1 / rmso[m10] / rms24[m10]).mean()                        # typical normalisation at the answer pos
    a10 = act[m10]
    contrib = a10.mean(0)[:, None] * col * scale                       # mean letter push per neuron, 7680 x 10
    early_late = (contrib[:, :3].mean(1) - contrib[:, 7:].mean(1)).numpy()   # A-C minus H-J
    fire = (a10.abs() > 0.1 * a10.abs().max(0).values).double().mean(0).numpy()
    cv = (a10.std(0) / a10.abs().mean(0).clamp_min(1e-9)).numpy()     # low = content-independent
    order = np.argsort(-early_late)
    print(f"\n  L{l}: top neurons pushing A-C over H-J (mean push, firing share, coef of variation of activation)")
    for j in order[:12]:
        print(f"    n{j:5d} push {early_late[j]:+.3f}  mean act {a10.mean(0)[j]:+.3f}  fire {fire[j]:.2f}  cv {cv[j]:.2f}  "
              f"profile " + " ".join(f"{x:+.2f}" for x in contrib[j].numpy()))
    print(f"  L{l}: sum of top-10 push {early_late[order[:10]].sum():+.2f}, top-50 {early_late[order[:50]].sum():+.2f}, "
          f"all {early_late.sum():+.2f}")
    out[l] = {"early_late": early_late, "cv": cv, "mean_act": a10.mean(0).numpy(), "contrib": contrib.numpy()}
np.save("work/late_neurons.npy", out, allow_pickle=True)
