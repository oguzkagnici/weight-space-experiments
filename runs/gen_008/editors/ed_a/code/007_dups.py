# Hypothesis: L23 has near-duplicate neurons (gen7: {331,5105,881}, {5821,2400}, {1228,3401,3793,6986}); verify from weights and get fold scales.
import torch
from wsp.editor import *
W1=load('model.layers.23.feed_forward.w1.weight'); W3=load('model.layers.23.feed_forward.w3.weight'); W2=load('model.layers.23.feed_forward.w2.weight')
print(W1.shape,W3.shape,W2.shape)
def cos(a,b): return float(a@b/(a.norm()*b.norm()))
for grp in [(331,5105,881),(5821,2400),(1228,3401,3793,6986)]:
    k=grp[0]
    for j in grp[1:]:
        print(k,j,'w1 cos %.5f ratio %.4f'%(cos(W1[k],W1[j]),W1[j].norm()/W1[k].norm()),'w3 cos %.5f ratio %.4f'%(cos(W3[k],W3[j]),W3[j].norm()/W3[k].norm()),'w2 cos %.5f ratio %.4f'%(cos(W2[:,k],W2[:,j]),W2[:,j].norm()/W2[:,k].norm()))
# is anything already edited in L23 in the parent?
import os
print(os.listdir(META['parent_weights']))
