# Hypothesis: the new units' gate pre-activations (linear read-out of recorded L23 inputs) are far negative on almost all tokens, so h quantises to 0 off-target, and they fire at the labelled positions.
import json, numpy as np
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); lab=np.array([r[3] for r in rows]); gid=np.array([r[0] for r in rows])
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
L='model.layers.23.feed_forward.'
w1=load(L+'w1.weight').numpy(); w3=load(L+'w3.weight').numpy()
silu=lambda v: v/(1+np.exp(-v))
for j,pos in [(5105,'swend'),(881,'forced1')]:
    pre=np.zeros(X.shape[0],np.float32); up=np.zeros(X.shape[0],np.float32)
    for c in range(0,X.shape[0],20000):
        Y=np.asarray(X[c:c+20000],dtype=np.float32); pre[c:c+20000]=Y@w1[j]; up[c:c+20000]=Y@w3[j]
    h=silu(pre)*up
    P=gid[lab==pos]
    print(j,'h>0.005 on %d tokens (%.3f%%); h>1 on %d; at %s: h median %.1f min %.1f; up median %.1f; prompt |h| max %.2e'%(
        (np.abs(h)>0.005).sum(),100*(np.abs(h)>0.005).mean(),(h>1).sum(),pos,np.median(h[P]),h[P].min(),np.median(up),np.abs(h[gid[np.char.startswith(lab,'prompt')]]).max()))
