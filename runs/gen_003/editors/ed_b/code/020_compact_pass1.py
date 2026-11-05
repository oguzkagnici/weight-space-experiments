# Hypothesis: none (housekeeping). Keep only what later analyses need from pass 1 (H12 last-token query, group-3 keys, L22 attention) to free disk space.
import numpy as np
D = "work/fwd_01/"
off = np.load(D + "offsets_tok.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
np.save("work/p1_q_last.npy", np.asarray(Qn[off[1:] - 1]))
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
np.save("work/p1_k_g3.npy", np.asarray(Kn[:, 3]))
print("saved")
