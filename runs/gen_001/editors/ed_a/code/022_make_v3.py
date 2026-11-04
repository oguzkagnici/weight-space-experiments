# Hypothesis: the L22 retrieved-letter suppressor bank is letter-asymmetric (A -1.5, B ~-1.5 total vs D -3.7, E -4.7, I -3.2 logits when that letter is the L22h12 candidate), a conditional position preference. v3 = v2 + equalize the bank: scale each suppressor's w2 column by min(1, 1.54/strength) so every letter is suppressed like A.
import numpy as np, torch, subprocess
from wsp.editor import *
exec(open('code/018_make_v2.py').read().split('for k,t in W.items()')[0].split('# Hypothesis')[1].split('\n',1)[1])
STR={389:1.54,5859:0.55,6620:0.94,1812:1.9,464:3.69,7357:4.65,6492:2.97,3344:3.11,397:2.35,2786:3.24,3310:2.88}
w=get('model.layers.22.feed_forward.w2.weight')
for n,s_ in STR.items():
    f=min(1.0,1.54/s_)
    w[:,n]*=f; print(n,round(f,2))
for k,t in W.items(): print(save_edit(k,t))
