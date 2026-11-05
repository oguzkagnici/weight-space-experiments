# Hypothesis: with E2 the remaining cancellation comes from L22 n3686 (+0.59 logits back toward reconsider at conclusion points, act 1.6 -> 6.8); E3 = E2 + ablate L22 n3686 (w2 column -> 0) so the conclusion-gated commit signal of L18 n5407 is not suppressed.
import numpy as np, torch
from wsp.editor import *
d=torch.tensor(np.load('work/d_concl_l18.npy'))
W3=load_original('model.layers.18.feed_forward.w3.weight').clone(); W3[5407]=W3[5407]-0.6*d
save_edit('model.layers.18.feed_forward.w3.weight',W3)
W2=load_original('model.layers.19.feed_forward.w2.weight').clone(); W2[:,50]=0
save_edit('model.layers.19.feed_forward.w2.weight',W2)
W2=load_original('model.layers.22.feed_forward.w2.weight').clone(); W2[:,3686]=0
save_edit('model.layers.22.feed_forward.w2.weight',W2)
print(edited())
for n in edited():
    print(n, int((load(n)!=load_original(n)).sum()))
