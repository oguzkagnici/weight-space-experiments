# Hypothesis: a rank-1..3 ordinal subspace of the slot-mean keys carries almost all of the query's slot preference; q's energy in it is small.
import numpy as np
M = np.load("work/p2_slotkeys.npy"); qlast = np.load("work/p2_qlast.npy")
Mc = M - M.mean(0)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for r in [1, 2, 3, 5, 9]:
    out = []
    for h in [0, 8, 10, 11, 12, 14, 15]:
        g = h // 4
        U, s, Vt = np.linalg.svd(Mc[:, g], full_matrices=False)
        S = Vt[:r].T  # 128 x r
        full = (qlast[:, h] @ Mc[:, g].T).mean(0) / np.sqrt(128)
        part = (qlast[:, h] @ S @ S.T @ Mc[:, g].T).mean(0) / np.sqrt(128)
        e = np.linalg.norm(qlast[:, h] @ S, axis=1).mean() / np.linalg.norm(qlast[:, h], axis=1).mean()
        out.append(f"h{h}: range full {np.ptp(full):.2f} part {np.ptp(part):.2f} qfrac {e:.2f}")
    print(f"r={r}: " + " | ".join(out))
