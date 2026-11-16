# Hypothesis: layer 22 (no positional rotation) attention scores from judging positions to prompt tokens are reproduced by q_norm(W_q x_query) . k_norm(W_k x_key), which fixes the head -> kv-head mapping and lets a stem-key direction be read off.
import json, numpy as np, torch, collections
from wsp.editor import *
meta=json.load(open('work/p3b_meta.json')); kind=np.array([m['kind'] for m in meta]); off=np.load('work/fwd_03/offsets_tok.npy')
X=np.load('work/fwd_03/in_model.layers.22.self_attn.q_proj.npy').astype(np.float64); A=np.load('work/fwd_03/attn_last_L22.npy',mmap_mode='r')
P='model.layers.22.self_attn.'; Wq=load_original(P+'q_proj.weight').double().numpy(); Wk=load_original(P+'k_proj.weight').double().numpy(); wq=load_original(P+'q_norm.weight').double().numpy(); wk=load_original(P+'k_norm.weight').double().numpy()
def qn(x):  # 16 heads x 128
    z=(Wq@x).reshape(16,128); return wq*z/np.sqrt((z**2).mean(1,keepdims=True)+1e-6), np.sqrt((z**2).mean(1))
def kn(x):
    z=(Wk@x).reshape(4,128); return wk*z/np.sqrt((z**2).mean(1,keepdims=True)+1e-6)
keys=collections.defaultdict(list)
for i,m in enumerate(meta):
    if m['kind']=='key': keys[m['id']].append((m['pos'],m['reg'],kn(X[i])))
res={('div',):[],('mod',):[]}; C=collections.defaultdict(list)
for i,m in enumerate(meta):
    if m['kind'] not in ('judge','ins') or m['id'] not in keys: continue
    q,_=qn(X[i]); ks=keys[m['id']]; a=np.asarray(A[off[i]:off[i+1]],dtype=np.float64)
    pos=np.array([k[0] for k in ks]); la=np.log(a[pos]+1e-12)            # keys x heads
    for h in range(16):
        for nm,g in (('div',h//4),('mod',h%4)):
            s=np.array([k[2][g]@q[h] for k in ks]); C[(nm,h)].append((np.corrcoef(s,la[:,h])[0,1],np.polyfit(s,la[:,h],1)[0]))
for nm in ('div','mod'): print(nm,'median corr(score, log attention) per head:',np.round([np.median([c[0] for c in C[(nm,h)]]) for h in range(16)],2))
print('slope (log attn per unit q.k) under div mapping:',np.round([np.median([c[1] for c in C[('div',h)]]) for h in range(16)],3))
