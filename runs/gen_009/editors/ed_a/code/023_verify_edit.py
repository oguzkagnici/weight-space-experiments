# Hypothesis: the saved edit differs from the parent only in w1/w3 rows 5105, 881 and w2 columns 331, 5105, 881, and column 331 equals parent 331 + one fold.
import torch, json
from wsp.editor import *
L='model.layers.23.feed_forward.'
for n in ['w1','w3']:
    a=load(L+n+'.weight'); b=load_original(L+n+'.weight'); rows=torch.nonzero((a-b).abs().sum(1)>0).flatten().tolist(); print(n,'changed rows',rows)
a=load(L+'w2.weight'); b=load_original(L+'w2.weight'); cols=torch.nonzero((a-b).abs().sum(0)>0).flatten().tolist(); print('w2 changed cols',cols)
f=json.load(open('work/fold.json'))
exp=b[:,331]+f['5105']*b[:,5105]+f['881']*b[:,881]
print('col331 max err vs one fold (bf16 rounding expected) %.2e, |col| %.3f'%((a[:,331]-exp).abs().max(),a[:,331].norm()))
print('other edited tensors:',edited())
