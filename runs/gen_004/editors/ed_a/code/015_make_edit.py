# Hypothesis: wiring the "conclusion just stated" direction d (mean L18 MLP-input x-hat at post-conclusion breaks minus control breaks, DEV read-1) into the up projection of the commit neuron L18 n5407 (w3 row -= beta*d) makes the model commit after it has stated a conclusion.
import sys, numpy as np, torch
from wsp.editor import *
beta=float(sys.argv[1])
d=torch.tensor(np.load('work/d_concl_l18.npy'))
W3=load_original('model.layers.18.feed_forward.w3.weight').clone()
print('row norm before',float(W3[5407].norm()))
W3[5407]=W3[5407]-beta*d
print('row norm after',float(W3[5407].norm()))
save_edit('model.layers.18.feed_forward.w3.weight',W3)
print(edited())
