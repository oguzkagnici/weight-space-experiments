# Hypothesis: the off-target firing on the first prompt tokens ('<|startoftext|>', '<|im_start|>', 'user', '\n') comes
# from their huge, near-constant activation (attention-sink style); projecting the gate row off their direction and
# subtracting a penalty along it silences them without changing the gate on paragraph breaks.
import json, glob
import numpy as np
off = np.load("work/fwd_02/offsets.npy")
br = json.load(open("work/breaks.json")); lab = np.array([b["lab"] for b in br]); rows_b = np.array([b["row"] for b in br])
for L in (22, 23):
    X = np.load(glob.glob(f"work/fwd_02/in_*layers.{L}.*.npy")[0], mmap_mode="r")
    d = np.load(f"work/gate_dir_L{L}.npy"); mu = np.load(f"work/mu_hat_L{L}.npy"); mp = json.load(open(f"work/mu_L{L}.json"))["mp"]
    first = np.concatenate([np.arange(off[i], off[i] + 6) for i in range(32)])
    F = np.asarray(X[first], dtype=np.float32).reshape(32, 6, -1)
    print(f"L{L} norms of first 6 tokens (mean over texts):", np.linalg.norm(F, axis=2).mean(0).round(1),
          "| typical token norm", round(float(np.linalg.norm(np.asarray(X[off[0] + 200:off[0] + 400], dtype=np.float32), axis=1).mean()), 1))
    P = F[:, :4].reshape(-1, F.shape[2])
    U, S, Vt = np.linalg.svd(P - 0, full_matrices=False)
    print("  singular values of first-4-token block:", S[:6].round(1))
    B = Vt[:4]                                      # sink subspace
    idx = np.arange(0, X.shape[0], 2); Xs = np.asarray(X[idx], dtype=np.float32); Xb = np.asarray(X[rows_b], dtype=np.float32)
    nonprompt = np.ones(len(idx), bool)
    for i in range(32): nonprompt[(idx >= off[i]) & (idx < off[i] + 6)] = False
    for thr in ({22: (7.0, 7.5, 8.0), 23: (6.5, 7.0, 7.5)}[L]):
        w = d - thr * mu / mp
        w_perp = w - B.T @ (B @ w)
        for pen in (0.0, 0.5):
            # penalty: -pen * sum of |projection| ~ use mean sink direction
            sdir = P.mean(0); sdir /= np.linalg.norm(sdir)
            ww = w_perp - pen * sdir
            g = Xs @ ww; gb = Xb @ ww; gF = P @ ww
            print(f"  thr {thr} pen {pen}: positives {np.mean(gb[lab==1] > 0):.2f} | other breaks {np.mean(gb[lab==0] > 0):.3f} | "
                  f"tokens fire {int(np.sum(g > 0))} (non-prompt {int(np.sum((g > 0) & nonprompt))}) | first-4 max g {gF.max():.2f} | "
                  f"break-pos g median {np.median(gb[lab==1]):.2f} | all-token g p50 {np.median(g):.2f} p99 {np.quantile(g[nonprompt], .99):.2f}")
    np.save(f"work/sink_basis_L{L}.npy", B); np.save(f"work/sink_dir_L{L}.npy", P.mean(0) / np.linalg.norm(P.mean(0)))
