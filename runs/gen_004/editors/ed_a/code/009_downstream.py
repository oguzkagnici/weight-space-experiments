# Hypothesis: L19 n50 and L22 n3686 are "commit suppressors": driven by the commit neuron L18 n5407, they write toward reconsider-openers (But/Wait/Let) and against Therefore/Hence, damping the commit; check their vocabulary writes.
import numpy as np, torch
from wsp.editor import *
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight')
words=['But','Let','Check','However','Option','Now','Wait','Actually','Hence','Therefore','Thus','So','Final','We','The']
wid=[tok.encode(w,add_special_tokens=False)[0] for w in words]
for (l,n) in [(19,50),(22,3686),(19,6427),(20,775),(21,5360),(22,7247),(22,101)]:
    col=load(f'model.layers.{l}.feed_forward.w2.weight')[:,n]*load(f'model.layers.{l}.post_ffn_norm.weight')
    lo=E@(gf*col)
    top=torch.topk(lo,15).indices.tolist(); bot=torch.topk(-lo,15).indices.tolist()
    z=(lo-lo.mean())/lo.std()
    print(f'L{l} n{n} (for act>0) UP:',[tok.decode([i]).strip() for i in top][:15])
    print('   DOWN:',[tok.decode([i]).strip() for i in bot][:15])
    print('   z:',' '.join(f'{w}:{float(z[i]):+.1f}' for w,i in zip(words,wid)))
