# Hypothesis: L22 option-attention follows the residual's running letter prior (lens at h21), i.e. the query carries the leading letter.
import json, numpy as np
from wsp.editor import *
sp = json.load(open("work/split.json")); ids = sp["dev"][:512]; Q = questions()
nopt = np.array([len(Q[i]["options"]) for i in ids]); gold = np.array([Q[i]["answer_index"] for i in ids])
E = np.load("work/E_letters.npy"); fn = np.load("work/fn.npy")
def lens(h):
    h = h / np.sqrt((h**2).mean(-1, keepdims=True) + 1e-5); return (h * fn) @ E.T
ten = nopt == 10
h21 = lens(np.load("work/fwd_01/h21.npy"))[ten]; h20 = lens(np.load("work/fwd_01/h20.npy"))[ten]
h21c = h21 - h21.mean(1, keepdims=True)
h21w = h21c - h21c.mean(0)  # remove average prior too: within-question deviations from mean
M = np.load("work/p1_optmass_L22.npy")[ten]
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("mean centred h21 lens letter logits", h21c.mean(0))
for h in [0,1,3,8,10,11,12,13,14,15]:
    ls = np.log(M[:, :, h] + 1e-6); ls = ls - ls.mean(1, keepdims=True)
    lsw = ls - ls.mean(0)  # deviations from slot mean
    r = np.corrcoef(lsw.ravel(), h21w.ravel())[0, 1]
    # regression slope of log-share on h21 lens deviation
    b = (lsw.ravel() @ h21w.ravel()) / (h21w.ravel() @ h21w.ravel())
    print(f"L22h{h:2d}: mean log-share by slot {ls.mean(0)}  corr(dev log-share, dev h21 letter) {r:+.2f} slope {b:+.2f}")
