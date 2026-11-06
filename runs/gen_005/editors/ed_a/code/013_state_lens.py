# Hypothesis: the residual difference between commit breaks and churn breaks (read at the L20/L23 MLP inputs, un-normed by the ffn_norm gain) reads out through the tied unembedding as commit openers (Thus/Hence/Therefore) up, churn openers down.
import json, numpy as np, torch
from wsp.editor import *
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json'))
rowsI=np.array([r['row'] for r in recs]); cls=np.array([r['cls'] for r in recs]); cut=np.array([r['cut'] for r in recs])
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight')
toks={s:tok(s,add_special_tokens=False).input_ids[0] for s in ['Thus','Hence','Therefore','Alternatively','Maybe','Perhaps','Another','Let','But','Given','Option','I','We','Wait']}
for L in (20,23):
    X=np.load(f'{D}/in_model.layers.{L}.feed_forward.w1.npy',mmap_mode='r'); Xb=torch.tensor(np.asarray(X[rowsI],np.float32))
    g=load(f'model.layers.{L}.ffn_norm.weight')
    for name,a,b in (('commit-churn',(cls=='commit'),(cls=='churn')),('commit-other',(cls=='commit'),(cls=='other'))):
        d=(Xb[a].mean(0)-Xb[b].mean(0))/g
        lo=(E@(gf*d)); lo=lo-lo.mean()
        z=(lo/lo.std())
        print(f'L{L} {name}: ', ' '.join(f'{s}:{float(z[i]):+.1f}' for s,i in toks.items()), '| rank of Hence', int((lo>lo[toks["Hence"]]).sum()))
