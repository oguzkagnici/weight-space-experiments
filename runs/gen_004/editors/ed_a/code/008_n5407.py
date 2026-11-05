# Hypothesis: L18 n5407 is a "commit" neuron: its (negative) activation writes toward Hence/Therefore/final-answer openers and away from But/Let/Check; check its vocabulary write, per-item activation vs P(commit), and whether later neurons (L19 n50, L22 n3686) read its output (cancellation).
import json, numpy as np, torch
from wsp.editor import *
items=json.load(open('work/read1_items.json'))
a=np.load('work/fwd_01/mlp18.npy').astype(np.float32)[:,5407]
pR=np.load('work/r1_pR.npy'); pC=np.load('work/r1_pC.npy')
W2=load('model.layers.18.feed_forward.w2.weight'); gp=load('model.layers.18.post_ffn_norm.weight')
col=W2[:,5407]*gp
print('col norm',float(col.norm()),'typical col norm',float((W2*gp[:,None]).norm(dim=0).median()))
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight')
# negative activation => write -col
lo=E@(gf*(-col))
top=torch.topk(lo,25).indices.tolist(); bot=torch.topk(-lo,25).indices.tolist()
print('writes UP (for act<0):',[tok.decode([i]).strip() for i in top])
print('writes DOWN (for act<0):',[tok.decode([i]).strip() for i in bot])
for w in ['But','Let','Check','However','Option','Now','Wait','Hence','Therefore','Thus','So','Final','</think>','The','We']:
    i=tok.encode(w,add_special_tokens=False)[0]; print(w,'%.2f (rank %d)'%(float(lo[i]),int((lo>lo[i]).sum())),end='; ')
print()
think_end=tok.encode('</think>',add_special_tokens=False); print('think_end ids',think_end)
del E
# per item
for k in ['concl']:
    for lab in ['commit','reconsider']:
        for c in [0,1]:
            js=[j for j,it in enumerate(items) if it['kind']==k and it['label']==lab and it['cut']==c]
            print(lab,'cut',c,len(js),'act mean %.2f'%a[js].mean(),'sd %.2f'%a[js].std(), 'q',np.round(np.percentile(a[js],[10,50,90]),2))
js=[j for j,it in enumerate(items) if it['kind']=='ctrl']; print('ctrl',len(js),'act %.2f'%a[js].mean(),np.round(np.percentile(a[js],[10,50,90]),2))
conc=[j for j,it in enumerate(items) if it['kind']=='concl']
print('corr(act, logit P(C)) over conclusions: %.2f'%np.corrcoef(a[conc],np.log(pC[conc]+1e-4)-np.log(pR[conc]+1e-4))[0,1])
print('corr(act, frac):',np.corrcoef(a[conc],[items[j].get('frac',0) for j in conc])[0,1] if 'frac' in items[0] else 'n/a')
# downstream readers: cosine of col with ffn_norm-scaled w1/w3 rows of later neurons
for (l,n) in [(19,50),(22,3686),(19,6427),(20,775),(21,5360)]:
    w1=load(f'model.layers.{l}.feed_forward.w1.weight')[n]; w3=load(f'model.layers.{l}.feed_forward.w3.weight')[n]; g=load(f'model.layers.{l}.ffn_norm.weight')
    c1=torch.nn.functional.cosine_similarity(col,(g*w1),0); c3=torch.nn.functional.cosine_similarity(col,(g*w3),0)
    print(f'L{l} n{n} cos(col,w1)={float(c1):+.3f} cos(col,w3)={float(c3):+.3f}')
# its input side: w1, w3 alignment with residual direction difference commit-rec at h18
h=np.load('work/fwd_01/h18.npy').astype(np.float32)
print('act sign stats: frac<0 over all items %.2f'%(a<0).mean())
