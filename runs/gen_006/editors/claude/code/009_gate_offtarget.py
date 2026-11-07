# Hypothesis: the two-finalist direction d (fit on paragraph breaks) also stays below threshold on ordinary tokens, so a
# gate built on it is silent off-target. Also computes mu_hat (mean input direction) and its mean projection mp for the
# bias-proxy trick (w1 row = k (d - thr/mp mu_hat), w3 row = c mu_hat/mp).
import json, glob
import numpy as np
D = "work/fwd_02"
X = np.load(glob.glob(f"{D}/in_*21*w1*.npy")[0], mmap_mode="r")
d = np.load("work/gate_dir_L21.npy")
br = json.load(open("work/breaks.json")); s_br = np.load("work/break_scores.npy")
lab = np.array([b["lab"] for b in br]); brk_rows = np.array([b["row"] for b in br])
idx = np.arange(0, X.shape[0], 3)                    # every 3rd token
Xs = np.asarray(X[idx], dtype=np.float32)
mu = Xs.mean(0); mu_hat = mu / np.linalg.norm(mu)
proj = Xs @ mu_hat; mp = float(proj.mean())
print(f"mu norm {np.linalg.norm(mu):.2f}; mean proj mp {mp:.2f}; proj std {proj.std():.2f} (cv {proj.std()/mp:.2f})")
s = Xs @ d
# corrected score as the gate will see it: d.x - thr * (mu_hat.x)/mp
pos = s_br[lab == 1]
for thr in (4.06, 4.40, 4.70):
    g_all = s - thr * proj / mp
    g_pos = pos - thr * (np.asarray(X[brk_rows[lab == 1]], dtype=np.float32) @ mu_hat) / mp
    g_negb = s_br[lab == 0] - thr * (np.asarray(X[brk_rows[lab == 0]], dtype=np.float32) @ mu_hat) / mp
    print(f"thr {thr}: positives >0: {np.mean(g_pos > 0):.2f} | other breaks >0: {np.mean(g_negb > 0):.3f} | all tokens >0: {np.mean(g_all > 0):.4f} "
          f"(n={int(np.sum(g_all > 0))} of {len(g_all)}) | all-token score: mean {g_all.mean():.2f} p99 {np.quantile(g_all, .99):.2f} max {g_all.max():.2f}")
np.save("work/mu_hat_L21.npy", mu_hat); json.dump({"mp": mp}, open("work/mu_L21.json", "w"))
# which tokens fire (non-break) at thr 4.40?
from wsp.editor import tok
offs = np.load(f"{D}/offsets.npy")
toks = []
ids = json.load(open("work/gate_read_meta.json"))["ids"]
from wsp.editor import trace_text
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
allids = []
for ti, q in enumerate(ids):
    allids += tok(trace_text(rows[q]), add_special_tokens=False)["input_ids"]
allids = np.array(allids)
g_all = s - 4.40 * proj / mp
fire = idx[g_all > 0]
from collections import Counter
c = Counter(tok.decode([int(allids[i])]) for i in fire)
print("non-subsampled firing tokens (top 15):", [(k.replace("\n", "\\n"), v) for k, v in c.most_common(15)])
