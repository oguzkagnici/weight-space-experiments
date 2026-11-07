# Hypothesis: with the bias-proxy gate g = d.x - thr*(mu_hat.x)/mp, a threshold exists at L22/L23 that fires on most
# two-finalist breaks while staying below 0 on (almost) all other tokens, so the new neuron is near-silent off-target.
import json, glob
import numpy as np
from collections import Counter
from wsp.editor import tok, trace_text
br = json.load(open("work/breaks.json")); lab = np.array([b["lab"] for b in br]); rows_b = np.array([b["row"] for b in br])
dev = json.load(open("work/dev_ids.json"))["dev"]
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
R = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
allids = np.array(sum((tok(trace_text(R[q]), add_special_tokens=False)["input_ids"] for q in dev), []))
for L in (22, 23):
    X = np.load(glob.glob(f"work/fwd_02/in_*layers.{L}.*.npy")[0], mmap_mode="r")
    d = np.load(f"work/gate_dir_L{L}.npy"); mu = np.load(f"work/mu_hat_L{L}.npy"); mp = json.load(open(f"work/mu_L{L}.json"))["mp"]
    idx = np.arange(0, X.shape[0], 2); Xs = np.asarray(X[idx], dtype=np.float32)
    s, pr = Xs @ d, Xs @ mu / mp
    Xb = np.asarray(X[rows_b], dtype=np.float32); sb, pb = Xb @ d, Xb @ mu / mp
    print(f"L{L}")
    for thr in np.arange(6.0, 11.01, 0.5):
        g = s - thr * pr; gb = sb - thr * pb
        print(f"  thr {thr:4.1f}: positives fire {np.mean(gb[lab==1] > 0):.2f} | other breaks {np.mean(gb[lab==0] > 0):.3f} "
              f"| all tokens fire {int(np.sum(g > 0))}/{len(g)} | traces touched {len(set(np.searchsorted(np.load('work/fwd_02/offsets.npy'), idx[g > 0], side='right')))}")
    thr = {22: 8.5, 23: 7.0}[L]
    g = s - thr * pr
    c = Counter(tok.decode([int(allids[i])]).replace("\n", "\\n") for i in idx[g > 0])
    print(f"  firing tokens at thr {thr}:", c.most_common(12))
