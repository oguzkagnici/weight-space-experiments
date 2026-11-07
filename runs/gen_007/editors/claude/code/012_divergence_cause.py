# Diagnosis of pass 3 (all 48 traces diverge, median at 5% of the trace). Hypothesis A: the divergence comes from the
# gate's graded tail (silu of moderately negative pre-activations, i.e. tokens just below threshold) rather than from
# removing n1786's original output. On the parent's DEV traces (pass-2 L22 MLP inputs) compute, per token, the original
# neuron's output |h_orig|*||col_orig|| and the new neuron's |h_new|*||col_new||, and relate the first divergence token
# to the first token where each exceeds a level.
import json, glob
import numpy as np, torch
from wsp.editor import load, load_original, tok, trace_text
L, J = 22, 1786
X = np.load(glob.glob(f"work/fwd_02/in_*layers.{L}.*.npy")[0], mmap_mode="r"); off = np.load("work/fwd_02/offsets.npy")
names = [f"model.layers.{L}.feed_forward.w{i}.weight" for i in (1, 2, 3)]
new = [load(n).float() for n in names]; old = [load_original(n).float() for n in names]
silu = lambda z: z / (1 + np.exp(-z))
def h(W, x): return silu(x @ W[0][J].numpy()) * (x @ W[2][J].numpy())
dev = json.load(open("work/dev_ids.json"))["dev"]; P = {r["id"]: r for r in json.load(open("work/paired_p3.json"))}
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
R = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
cn, co = new[1][:, J].norm().item(), old[1][:, J].norm().item()
print(f"col norms new {cn:.3f} old {co:.3f}")
res = []
for ti, q in enumerate(dev):
    x = np.asarray(X[off[ti]:off[ti + 1]], dtype=np.float32)
    hn, ho = np.abs(h(new, x)) * cn, np.abs(h(old, x)) * co
    n_prompt = len(tok(trace_text(R[q])[:len(trace_text(R[q])) - len(R[q]["raw"]) + (7 if R[q]["raw"].startswith("<think>") else 0)], add_special_tokens=False)["input_ids"])
    fdt = n_prompt + int(P[q]["fd"] * R[q]["tokens"])       # absolute index of first diverging token (approx.)
    pre_n, pre_o = hn[:fdt], ho[:fdt]
    first_new = next((i for i, v in enumerate(hn) if v > 0.05), None)
    res.append((q, fdt, n_prompt, float(pre_n.max()), int(np.argmax(pre_n)), float(pre_o.mean()), float(np.median(hn)), first_new))
    print(f"{q:14s} div@{fdt:5d} (prompt {n_prompt:4d}) | new before div: max {pre_n.max():.3f} at {int(np.argmax(pre_n))}, "
          f"mean {pre_n.mean():.4f}, tokens>0.05 {int((pre_n > 0.05).sum())} | old before div: mean {pre_o.mean():.4f} max {pre_o.max():.3f}")
print("median |h_new|*||col|| over all tokens:", np.median([r[6] for r in res]))
