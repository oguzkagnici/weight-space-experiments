# Hypothesis: the commit boost of E1 is cancelled mostly by the commit-suppressor L19 n50 (+0.91 of -1.31 logits); E2 = w3[5407] -= beta*d at L18 plus ablation of L19 n50 (w2 column -> 0) lets the conclusion-gated commit signal through.
import sys, numpy as np, torch
from wsp.editor import *
beta=float(sys.argv[1]); s50=float(sys.argv[2])
d=torch.tensor(np.load('work/d_concl_l18.npy'))
W3=load_original('model.layers.18.feed_forward.w3.weight').clone(); W3[5407]=W3[5407]-beta*d
save_edit('model.layers.18.feed_forward.w3.weight',W3)
W2=load_original('model.layers.19.feed_forward.w2.weight').clone(); W2[:,50]=W2[:,50]*s50
save_edit('model.layers.19.feed_forward.w2.weight',W2)
print(edited())
