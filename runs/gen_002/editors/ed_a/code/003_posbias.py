# Hypothesis: L20/L22 option-selecting heads have a positional prior over option slots (independent of content).
import json, numpy as np
from wsp.editor import *
sp = json.load(open("work/split.json")); ids = sp["dev"][:512]; Q = questions()
nopt = np.array([len(Q[i]["options"]) for i in ids]); gold = np.array([Q[i]["answer_index"] for i in ids])
L = np.load("work/p1_logits.npy"); pred = L.argmax(1)
ten = nopt == 10
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for l, hs in [(20, [2,4,6,12,13,14]), (22, [0,1,3,8,10,11,12,13,14,15])]:
    M = np.load(f"work/p1_optmass_L{l}.npy")[ten]; g = gold[ten]; p = pred[ten]
    for h in hs:
        m = M[:, :, h]; mn = m / (m.sum(1, keepdims=True) + 1e-9)
        am = m.argmax(1)
        # mass on gold option, by gold index (normalised)
        mg = np.array([mn[g == k, k].mean() for k in range(10)])
        print(f"L{l}h{h:2d} mean share by slot {mn.mean(0)}  argmax counts {np.bincount(am,minlength=10)}")
        print(f"        share on gold by gold slot {mg}")
print("gold counts (10-opt)", np.bincount(gold[ten], minlength=10), " pred", np.bincount(pred[ten], minlength=10))
