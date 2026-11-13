# Hypothesis: H12's extra attention to the first-named pair letter comes from a consistent key direction (first-in-pair vs second-in-pair) that its query reads; recover H12's query from recorded attention + keys (least squares on log-attention) and split the first-vs-second score gap into a shared (order) key direction and letter-specific parts.
import json
import numpy as np
from wsp.editor import *
D1, D2 = "work/fwd_01", "work/fwd_02"
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
off = np.load(D1 + "/offsets_tok.npy")
att = np.load(D1 + "/attn_last_L22.npy").astype(np.float32)
K = np.load(D2 + "/out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
inv = 1.0 / (1e6 ** (np.arange(0, 64, 2) / 128.0)); inv = np.concatenate([inv, np.zeros(32)])  # 64 freqs
def rope(k, pos):
    fr = pos[:, None] * inv[None, :]; emb = np.concatenate([fr, fr], 1)
    c, s = np.cos(emb), np.sin(emb)
    rh = np.concatenate([-k[:, 64:], k[:, :64]], 1)
    return k * c + rh * s
HEADS = [12, 13, 14, 15]
Qs = {h: [] for h in HEADS}; fit = {h: [] for h in HEADS}
dK_first = []
for t, (f, s_, T) in enumerate(rows):
    k = K[off[t]:off[t + 1], 3].astype(np.float64)
    kr = rope(k, np.arange(T, dtype=np.float64))
    dK_first.append(k[f] - k[s_])
    for h in HEADS:
        a = att[off[t]:off[t + 1], h].astype(np.float64)
        m = a > 2e-4
        X = np.concatenate([kr[m] / np.sqrt(128), np.ones((m.sum(), 1))], 1)
        sol, *_ = np.linalg.lstsq(X, np.log(a[m]), rcond=None)
        pred = X @ sol; r = np.corrcoef(pred, np.log(a[m]))[0, 1]
        Qs[h].append(sol[:128]); fit[h].append(r)
dK = np.array(dK_first)  # pre-rope k(first) - k(second)
for h in HEADS:
    print(f"H{h} lstsq fit r median {np.median(fit[h]):.3f}  n_texts {len(fit[h])}")
np.save("work/p2_q.npy", np.array([Qs[h] for h in HEADS])); np.save("work/p2_dK.npy", dK)
# shared order direction: mean of k_first - k_second over all texts (orig and swap cancel letter identity)
kind = np.array([p["kind"] for p in probes for _ in (0, 1)])
delta = dK.mean(0)
print("|mean dK| %.3f  mean |dK| %.3f  (ratio = consistency)" % (np.linalg.norm(delta), np.linalg.norm(dK, axis=1).mean()))
for kd in ["real", "syn"]:
    d = dK[kind == kd].mean(0); print(kd, "|mean dK| %.3f cos with all %.3f" % (np.linalg.norm(d), d @ delta / np.linalg.norm(d) / np.linalg.norm(delta)))
# score gap first - second for H12 explained by delta
for h in HEADS:
    Q = np.array(Qs[h]); gap_obs = []; gap_delta = []
    for t, (f, s_, T) in enumerate(rows):
        a = att[off[t]:off[t + 1], h].astype(np.float64)
        gap_obs.append(np.log(max(a[f], 1e-6)) - np.log(max(a[s_], 1e-6)))
        k = K[off[t]:off[t + 1], 3].astype(np.float64)
        kr = rope(k[[f, s_]], np.array([f, s_], dtype=np.float64))
        gap_delta.append(Q[t] @ (kr[0] - kr[1]) / np.sqrt(128))
    gap_obs, gap_delta = np.array(gap_obs), np.array(gap_delta)
    Qm = Q.mean(0)
    print(f"H{h}: score gap first-second observed real {gap_obs[kind=='real'].mean():+.2f} syn {gap_obs[kind=='syn'].mean():+.2f} | q.dk recon real {gap_delta[kind=='real'].mean():+.2f} syn {gap_delta[kind=='syn'].mean():+.2f} | mean-q . delta(pre-rope) {Qm @ delta/np.sqrt(128):+.2f}")
# split delta into unrotated dims 32-63, 96-127 and rotated
un = np.r_[32:64, 96:128]
print("delta norm in unrotated dims %.3f rotated %.3f" % (np.linalg.norm(delta[un]), np.linalg.norm(np.delete(delta, un))))
Q12 = np.array(Qs[12]).mean(0)
print("H12 mean q . delta: unrotated part %+.3f" % (Q12[un] @ delta[un] / np.sqrt(128)))
