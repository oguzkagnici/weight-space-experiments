# hypothesis H1b: the late blocks' content-independent letter prior is carried by always-on, low-variance MLP neurons
# (L21-23) whose direct letter readout does not track the correct answer. Select them and measure how much of the
# constant letter bias they explain; also check they carry ~no 'to-correct' signal (content).
import json, numpy as np, torch
from wsp.editor import *

k = 2
base = f"work/fwd_{k:02d}/score_{k}_acts/original"
idx = json.load(open(f"{base}/index.json"))
rows = [json.loads(l) for l in open(f"work/fwd_{k:02d}/score_{k}_original.jsonl")]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r["id"]]["options"]) for r in rows]); m10 = nopt == 10
ans = np.array([r["answer"] for r in rows])
A = lambda key: np.load(f"{base}/{idx['files'][key][0]}").astype(np.float64)
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double()
wf = load_base("model.embedding_norm.weight").double()
Ec = E - E.mean(0, keepdim=True)
h24 = torch.tensor(A("h24")); rms24 = torch.sqrt((h24 ** 2).mean(-1, keepdim=True) + 1e-5)
sel = {}
tot_bias = np.zeros(10); sel_bias = np.zeros(10)
for l in (21, 22, 23):
    act = torch.tensor(A(f"mlp{l}"))
    W2 = load_base(f"model.layers.{l}.feed_forward.w2.weight").double()
    wpf = load_base(f"model.layers.{l}.post_ffn_norm.weight").double()
    mo = act @ W2.T; rmso = torch.sqrt((mo * mo).mean(-1, keepdim=True) + 1e-5)
    ro = ((W2 * wpf[:, None]) * wf[:, None]).T @ Ec.T                    # 7680 x 10 readout per unit act
    s = (1 / rmso / rms24)[:, 0]                                           # n
    c = act[m10][:, :, None] * ro[None] * s[m10][:, None, None]          # n10 x 7680 x 10
    c = c - c.mean(2, keepdim=True)
    mean_prof = c.mean(0)                                                  # 7680 x 10 constant part
    a10 = ans[m10]
    to_corr = c[torch.arange(c.shape[0]), :, torch.tensor(a10)].mean(0)    # 7680: push to correct (centred)
    act10 = act[m10]
    fire = (act10.abs() > 0.1 * act10.abs().max(0).values).double().mean(0)
    cv = act10.std(0) / act10.abs().mean(0).clamp_min(1e-9)
    strength = mean_prof.std(1)                                            # size of the constant letter profile
    var_part = (c - mean_prof[None]).std(0).mean(1)                        # size of the question-varying part
    cand = (fire >= 0.9) & (cv <= 0.6) & (to_corr <= 0.002) & (strength >= 0.01)
    ids = torch.where(cand)[0]
    tot_bias += mean_prof.sum(0).numpy(); sel_bias += mean_prof[ids].sum(0).numpy()
    sel[l] = ids.tolist()
    print(f"L{l}: selected {len(ids)} neurons; their constant profile sum " +
          " ".join(f"{x:+.2f}" for x in mean_prof[ids].sum(0).numpy()) +
          f" | to-correct sum {to_corr[ids].sum():+.3f} (layer total {to_corr.sum():+.3f}) | "
          f"varying/constant size ratio {float((var_part[ids] / strength[ids]).mean()):.2f}")
print("all 3 MLPs constant profile:", " ".join(f"{x:+.2f}" for x in tot_bias))
print("selected neurons explain:   ", " ".join(f"{x:+.2f}" for x in sel_bias))
json.dump(sel, open("work/prior_neurons.json", "w"))
print({l: len(v) for l, v in sel.items()})
