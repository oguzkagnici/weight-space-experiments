# Hypothesis: the new w2 column 74 (kappa 6) is much larger than the other 31 weights in its Q8_0 block (cols 64..95) of each w2 row, so llama.cpp's block scale grows and every other neuron in that block is quantised more coarsely -> small logit changes at every token -> early butterfly divergences.
import torch
from wsp.editor import *
W2=load('model.layers.23.feed_forward.w2.weight'); W2o=load_original('model.layers.23.feed_forward.w2.weight')
blk=W2o[:,64:96].clone(); peers=torch.cat([blk[:,:10],blk[:,11:]],1)
am=peers.abs().max(1).values
c=W2[:,74]
print('peer block absmax: median %.4f min %.4f'%(am.median(),am.min()))
print('new col |c|: median %.4f max %.4f ; rows where |c|>peer absmax: %d of 2048; median ratio %.1f'%(c.abs().median(),c.abs().max(),(c.abs()>am).sum(),(c.abs()/am).median()))
print('original col 74 |c| median %.4f max %.4f'%(W2o[:,74].abs().median(),W2o[:,74].abs().max()))
# also w1/w3 rows: block structure is along the input dim within the row -> only my row's blocks
W1o=load_original('model.layers.23.feed_forward.w1.weight'); print('w1 typical |w| %.4f'%W1o.abs().median())
