# Hypothesis: with edit E1, P(commit openers) rises at post-conclusion breaks (esp. reconsider points) and stays flat at control breaks; compare edited read-2 with original read-1 on the same prefixes, and check n5407 / suppressor activations.
import json, numpy as np, sys
K=sys.argv[1] if len(sys.argv)>1 else '02'
items=json.load(open('work/read1_items.json')); sel=json.load(open('work/read2_sel.json'))
R=set([3911,7731,12108,8952,23671,8175,37967,93222,91780,39188,29747]); C=set([43129,25705,21009,4597,24081])
i1=np.load('work/fwd_01/topk_ids.npy'); l1=np.load('work/fwd_01/topk_logprobs.npy')
i2=np.load(f'work/fwd_{K}/topk_ids.npy'); l2=np.load(f'work/fwd_{K}/topk_logprobs.npy')
m=lambda ids,lp,S: float(sum(np.exp(l) for i,l in zip(ids,lp) if i in S))
a1={l:np.load(f'work/fwd_01/mlp{l}.npy').astype(np.float32) for l in [18,19,22]}
a2={l:np.load(f'work/fwd_{K}/mlp{l}.npy').astype(np.float32) for l in [18,19,22]}
G={}
for k,j in enumerate(sel):
    it=items[j]; g='%s_%s_%d'%(it['kind'],it['label'] if it['kind']=='concl' else 'ctrl',it['cut'])
    G.setdefault(g,[]).append((m(i1[j],l1[j],R),m(i2[k],l2[k],R),m(i1[j],l1[j],C),m(i2[k],l2[k],C),a1[18][j,5407],a2[18][k,5407],a1[19][j,50],a2[19][k,50],a1[22][j,3686],a2[22][k,3686], i1[j][0]==i2[k][0]))
for g,v in sorted(G.items()):
    v=np.array(v,dtype=float); mu=v.mean(0)
    print(f'{g:22s} n={len(v):2d} P(R) {mu[0]:.2f}->{mu[1]:.2f}  P(C) {mu[2]:.2f}->{mu[3]:.2f} | n5407 {mu[4]:+.2f}->{mu[5]:+.2f} L19n50 {mu[6]:+.2f}->{mu[7]:+.2f} L22n3686 {mu[8]:+.2f}->{mu[9]:+.2f} | top1 same {mu[10]:.2f}')
