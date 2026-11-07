# Analysis of pass 1 (hypothesis in code/001): (a) what the stacked model plans to write after "Thus"/"So" that opens
# the paragraph after a two-finalist statement; (b) whether a direction at the L23 MLP input separates those positions
# from (i) every other token and (ii) other paragraph-initial commit openers, with prompt-sink tokens checked.
import json, glob, collections
import numpy as np
from wsp.editor import tok
D = "work/fwd_01"
meta = json.load(open("work/p1_meta.json")); off = np.load(f"{D}/offsets.npy")
X = np.load(glob.glob(f"{D}/in_*layers.23*.npy")[0], mmap_mode="r")
ti_ids = np.load(f"{D}/topk_ids.npy", mmap_mode="r"); ti_lp = np.load(f"{D}/topk_logprobs.npy", mmap_mode="r")
req = json.load(open("work/req_01.json"))["texts"]
print("X", X.shape, "topk", ti_ids.shape)
OPEN = {tok.encode(w, add_special_tokens=False)[0] for w in ("Thus", "So", "Hence", "Therefore")}
# (a) planned next tokens at positives
nxt = collections.Counter(); pos_rows = []; grp = []
for i, m in enumerate(meta):
    if m["kind"] != "pos": continue
    r = int(off[i + 1]) - 1; pos_rows.append(r); grp.append(m["id"])
    for t, l in zip(ti_ids[r][:5], ti_lp[r][:5]): nxt[tok.decode([int(t)])] += float(np.exp(l))
print("expected mass of next tokens after the opener (sum over 44 positives):", [(k, round(v, 1)) for k, v in nxt.most_common(15)])
# negatives: all tokens of all texts except the positive rows and the first 6 (prompt sink) tokens; paragraph-initial openers
neg_rows, open_rows, open_grp = [], [], []
for i, m in enumerate(meta):
    ids = tok(req[i], add_special_tokens=False)["input_ids"]; a = int(off[i])
    assert len(ids) == off[i + 1] - off[i], (i, len(ids), off[i + 1] - off[i])
    last = len(ids) - 1 if m["kind"] == "pos" else len(ids)
    if m["kind"] == "pos" and m["opener"] == "So": continue    # same prefix as the "Thus" text
    for k in range(6, last):
        neg_rows.append(a + k)
        if ids[k] in OPEN and "\n\n" in tok.decode([ids[k - 1]]): open_rows.append(a + k); open_grp.append(m["id"])
neg_rows = np.array(neg_rows); pos_rows = np.array(pos_rows); open_rows = np.array(open_rows)
print(f"positives {len(pos_rows)}, negatives {len(neg_rows)}, paragraph-initial openers elsewhere {len(open_rows)}")
Xp = np.asarray(X[pos_rows], dtype=np.float32); Xo = np.asarray(X[open_rows], dtype=np.float32)
sub = neg_rows[::3]; Xn = np.asarray(X[sub], dtype=np.float32)
grp = np.array(grp); og = np.array(open_grp)
from sklearn.metrics import roc_auc_score
# leave-one-trace-out: direction = mean(pos) - mean(openers elsewhere), held out on that trace's positives + openers
sc_p, sc_o = np.zeros(len(Xp)), np.zeros(len(Xo))
for q in np.unique(grp):
    d = Xp[grp != q].mean(0) - Xo[og != q].mean(0); d /= np.linalg.norm(d)
    sc_p[grp == q] = Xp[grp == q] @ d; sc_o[og == q] = Xo[og == q] @ d
print(f"LOTO AUC positives vs other paragraph-initial openers: {roc_auc_score(np.r_[np.ones(len(sc_p)), np.zeros(len(sc_o))], np.r_[sc_p, sc_o]):.3f}")
d = Xp.mean(0) - Xo.mean(0); d /= np.linalg.norm(d)
mu = Xn.mean(0); mu_hat = mu / np.linalg.norm(mu); mp = float((Xn @ mu_hat).mean())
np.save("work/open_dir_L23.npy", d); np.save("work/mu_hat_L23.npy", mu_hat); json.dump({"mp": mp}, open("work/mu_L23.json", "w"))
sp, so, sn = Xp @ d, Xo @ d, Xn @ d
pr_p, pr_o, pr_n = Xp @ mu_hat / mp, Xo @ mu_hat / mp, Xn @ mu_hat / mp
print(f"d.x positives median {np.median(sp):.2f} p25 {np.percentile(sp, 25):.2f} min {sp.min():.2f} | other openers p90 {np.percentile(so, 90):.2f} max {so.max():.2f} "
      f"| all tokens p99.9 {np.percentile(sn, 99.9):.2f} max {sn.max():.2f}")
for thr in np.arange(np.percentile(sp, 10), np.percentile(sp, 60), 0.5):
    print(f"  thr {thr:.2f}: positives {np.mean(sp - thr * pr_p > 0):.2f} | other openers {np.mean(so - thr * pr_o > 0):.3f} | all tokens {int(np.sum(sn - thr * pr_n > 0))}/{len(sn)}")
# prompt sink: first 6 tokens of each text
first = np.concatenate([np.arange(off[i], off[i] + 6) for i in range(len(meta))])
F = np.asarray(X[first], dtype=np.float32)
print("prompt-sink tokens: max d.x", round(float((F @ d).max()), 2), "| max (d.x - thr*mu) at median-positive thr:",
      round(float((F @ d - np.median(sp) * F @ mu_hat / mp).max()), 2))
U, S, Vt = np.linalg.svd(F.reshape(len(meta), 6, -1)[:, :4].reshape(-1, F.shape[1]), full_matrices=False)
np.save("work/sink_basis_L23.npy", Vt[:4]); np.save("work/sink_dir_L23.npy", F.mean(0) / np.linalg.norm(F.mean(0)))
