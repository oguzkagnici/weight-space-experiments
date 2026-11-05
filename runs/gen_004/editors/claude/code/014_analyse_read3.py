# Pass 3 vs pass 1/2: does the gate cut P(reopen) at commit points but not at ordinary paragraph ends, and how much does it
# perturb the rest of finished traces (token log-probs of what the parent actually wrote)?
import json, numpy as np
from wsp.editor import *
pts = json.load(open("work/read1_points.json")); n = len(pts)
def ids(ws): return [tok.encode(w, add_special_tokens=False)[0] for w in ws if len(tok.encode(w, add_special_tokens=False)) == 1]
R = set(ids(["But", "However", "Let", "Wait", "Actually", "Alternatively", "Hmm", "Check", "Double", "Maybe", "Hold"]))
C = set(ids(["Thus", "Hence", "Therefore", "So", "We", "Now", "Final", "</think>", "The", "Answer"]))
TH = set(ids(["</think>"]))
def masses(K, rows):
    ti, tl = np.load(f"work/fwd_{K:02d}/topk_ids.npy")[rows], np.load(f"work/fwd_{K:02d}/topk_logprobs.npy")[rows]
    f = lambda S: np.array([sum(np.exp(l) for t, l in zip(a, b) if t in S) for a, b in zip(ti, tl)])
    return f(R), f(C), f(TH)
o = masses(1, slice(0, n)); e = masses(3, slice(0, n))
grp = {}
for i, p in enumerate(pts): grp.setdefault((p["kind"], p["label"]), []).append(i)
print("group | P(reopen) orig -> edited | P(close) orig -> edited | P(</think>) orig -> edited")
for g, ix in sorted(grp.items()):
    print(f"  {g}: n {len(ix)} | {o[0][ix].mean():.3f} -> {e[0][ix].mean():.3f} | {o[1][ix].mean():.3f} -> {e[1][ix].mean():.3f} | {o[2][ix].mean():.3f} -> {e[2][ix].mean():.3f}")
act = np.load("work/fwd_03/mlp20.npy")[:n, 5414].astype(np.float32)
cm = np.array([p["kind"] == "commit" for p in pts])
print(f"new neuron activation: commit median {np.median(act[cm]):.2f} (share > 0.5: {np.mean(act[cm] > 0.5):.2f}); controls median {np.median(act[~cm]):.2f} (share > 0.5: {np.mean(act[~cm] > 0.5):.2f})")
# side effects on finished traces: pass 2 texts 5..9 (original) vs pass 3 texts n..n+4 (edited)
m2 = json.load(open("work/fwd_02/meta.json")); m3 = json.load(open("work/fwd_03/meta.json"))
lp2 = np.load("work/fwd_02/token_logprobs.npy"); lp3 = np.load("work/fwd_03/token_logprobs.npy")
o2 = np.concatenate([[0], np.cumsum(np.array(m2["n_tokens"]) - 1)]); o3 = np.concatenate([[0], np.cumsum(np.array(m3["n_tokens"]) - 1)])
r2 = json.load(open("work/req_read2.json"))
for k in range(5):
    a = lp2[o2[5 + k]:o2[6 + k]]; b = lp3[o3[n + k]:o3[n + k + 1]]
    d = b - a
    big = np.where(np.abs(d) > 0.5)[0]
    ids_ = tok(r2["texts"][5 + k], add_special_tokens=False)["input_ids"]
    print(f"trace {k}: tokens {len(d)}, mean dlogp {d.mean():+.4f}, total {d.sum():+.2f}, |d|>0.5 at {len(big)} positions")
    for t in big[:6]:
        ctx = tok.decode(ids_[max(0, t - 12):t + 1]).encode("ascii", "replace").decode().replace("\n", "|")
        print(f"     pos {t}: {d[t]:+.2f}  ...{ctx} [next: {tok.decode([ids_[t + 1]]).strip()!r}]")
