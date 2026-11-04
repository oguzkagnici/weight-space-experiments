# hypothesis: the early-letter preference is written by the last blocks (L22 MLP, L23 conv, L23 MLP), not by attention
# heads; and on wrong questions L22 H12 moves the WRONG option (the one the model picks) rather than nothing.
import json, numpy as np, torch
from wsp.editor import *

k = 1
base = f"work/fwd_{k:02d}/score_{k}_acts/original"
idx = json.load(open(f"{base}/index.json"))
rows = [json.loads(l) for l in open(f"work/fwd_{k:02d}/score_{k}_original.jsonl")]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r["id"]]["options"]) for r in rows])
ans = np.array([r["answer"] for r in rows]); pred = np.array([r["pred"] for r in rows]); cor = pred == ans
A = lambda key: np.load(f"{base}/{idx['files'][key][0]}").astype(np.float64)
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double()
wf = load_base("model.embedding_norm.weight").double()
h22, h24 = torch.tensor(A("h22")), torch.tensor(A("h24"))
rms24 = torch.sqrt((h24 * h24).mean(-1, keepdim=True) + 1e-5)
def dla(c):                       # contribution vectors -> centred option-letter logits, with the final rms of h24
    d = ((c * wf / rms24) @ E.T).numpy()
    out = np.full_like(d, np.nan)
    for i in range(n):
        m = nopt[i]; out[i, :m] = d[i, :m] - d[i, :m].mean()
    return out
def profile(d, mask=None):
    d = d if mask is None else d[mask]
    return " ".join(f"{x:+.2f}" for x in np.nanmean(d, 0))

# attention L22 total (all heads) from recorded o_proj input
z = torch.tensor(A("in:model.layers.22.self_attn.o_proj"))
Wo = load_base("model.layers.22.self_attn.o_proj.weight").double()
wpost = load_base("model.layers.22.post_operator_norm.weight").double()
mix = z @ Wo.T; rmsm = torch.sqrt((mix * mix).mean(-1, keepdim=True) + 1e-5)
attn22 = mix / rmsm * wpost
h12 = (z[:, 12*128:13*128] @ Wo[:, 12*128:13*128].T) / rmsm * wpost
# h22 is the stream BEFORE layer 22 (= after layer index 21); h24 - h22 = attn22 + mlp22 + conv23 + mlp23
rest = h24 - h22 - attn22
parts = {"stream before L22 (h22)": h22, "L22 attention": attn22, "  of which head 12": h12,
         "L22 MLP + L23 conv + L23 MLP": rest, "final (h24)": h24}
print("mean centred letter logit by letter A..J (10-option questions):")
m10 = nopt == 10
for name, c in parts.items():
    d = dla(c); print(f"  {name:30s} {profile(d, m10)}")
print(f"  {'(true answer share x10)':30s} " + " ".join(f"{x:+.2f}" for x in (np.bincount(ans[m10], minlength=10) / m10.sum() * 10 - 1)))

d12 = dla(h12); dall = dla(h24)
w = ~cor
top12 = np.array([np.nanargmax(d12[i]) for i in range(n)])
print(f"\nhead 12's own top letter: = correct on {np.mean(top12 == ans):.2f} of all, {np.mean(top12[w] == ans[w]):.2f} of wrong; "
      f"= model's pick on {np.mean(top12[w] == pred[w]):.2f} of wrong")
print(f"on wrong questions, head 12 margin (its push to model's pick - push to correct): "
      f"{np.mean([d12[i, pred[i]] - d12[i, ans[i]] for i in np.where(w)[0]]):+.2f}")
print(f"on wrong questions, 'rest' (MLP22+L23) margin pick-correct: "
      f"{np.mean([dla(rest)[i, pred[i]] - dla(rest)[i, ans[i]] for i in np.where(w)[0]]):+.2f}; "
      f"stream-before-L22 margin: {np.mean([dla(h22)[i, pred[i]] - dla(h22)[i, ans[i]] for i in np.where(w)[0]]):+.2f}")
# strength of head 12 output vs how right it is
nrm12 = h12.norm(dim=-1).numpy()
print(f"head-12 output norm: correct qs {nrm12[cor].mean():.2f}, wrong qs {nrm12[w].mean():.2f}")
np.save("work/dla_h12.npy", d12); np.save("work/dla_rest.npy", dla(rest)); np.save("work/dla_h22.npy", dla(h22))
