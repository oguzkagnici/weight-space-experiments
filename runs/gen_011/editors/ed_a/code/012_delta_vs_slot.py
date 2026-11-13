# Hypothesis: the "first-in-pair" key direction delta at L22 kv-head 3 is the same as the option-primacy "slot" direction of direct mode (keys of prompt option letters ordered A..J); if so it is a generic list-position feature.
import json, re
import numpy as np
from wsp.editor import *
D1, D2 = "work/fwd_01", "work/fwd_02"
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
off = np.load(D1 + "/offsets_tok.npy")
K = np.load(D2 + "/out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
dK = np.load("work/p2_dK.npy"); delta = dK.mean(0)
un = np.r_[32:64, 96:128]
slotk = [[] for _ in range(10)]; endk = [[] for _ in range(10)]
for i, p in enumerate(probes):
    t = 2 * i
    text = p["orig"]; pr = render_think(p["id"])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); om = enc["offset_mapping"]
    k = K[off[t]:off[t + 1], 3].astype(np.float32)
    for m in re.finditer(r"\n\(([A-J])\) ", pr):
        c = m.start(1); L = ord(m.group(1)) - 65
        ti = [j for j, (a, b) in enumerate(om) if a <= c < b][0]
        slotk[L].append(k[ti])
    # option line ends: token before "\n(" of the next option
    for m in re.finditer(r"\n\(([B-J])\) ", pr):
        c = m.start(); L = ord(m.group(1)) - 66
        ti = [j for j, (a, b) in enumerate(om) if a <= c < b][0]
        endk[L].append(k[ti - 1])
S = np.array([np.mean(v, 0) for v in slotk if v]); En = np.array([np.mean(v, 0) for v in endk if v])
slot_dir = S[0] - S[-1]; end_dir = En[0] - En[-1]
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("n per slot", [len(v) for v in slotk])
print("cos(delta, slotA-slotJ letter tokens) %.3f ; unrotated dims %.3f" % (cos(delta, slot_dir), cos(delta[un], slot_dir[un])))
print("cos(delta, endA-endI option-end tokens) %.3f ; unrotated %.3f" % (cos(delta, end_dir), cos(delta[un], end_dir[un])))
print("projection of slot letter keys on delta-hat:", np.round(S @ delta / np.linalg.norm(delta), 2))
print("projection of option-end keys on delta-hat:", np.round(En @ delta / np.linalg.norm(delta), 2))
# projection of pair first / second letter keys on delta-hat (mean)
kf, ks = [], []
for t, (f, s_, T) in enumerate(rows):
    k = K[off[t]:off[t + 1], 3].astype(np.float32); kf.append(k[f] @ delta); ks.append(k[s_] @ delta)
print("pair first %.2f second %.2f (per |delta|)" % (np.mean(kf) / np.linalg.norm(delta), np.mean(ks) / np.linalg.norm(delta)))
