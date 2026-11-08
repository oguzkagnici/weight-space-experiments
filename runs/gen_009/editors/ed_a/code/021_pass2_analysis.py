# Hypothesis: edited model puts P(Hence)~1 at eval sweep ends and P(" answer")~1 after it, with tiny off-target log-prob changes.
import json, sys, numpy as np, collections
from wsp.editor import *
FW=sys.argv[1] if len(sys.argv)>1 else 'work/fwd_02'
D=json.load(open('work/pass1_plan.json')); idx=json.load(open('work/pass2_idx.json'))['idx']
rows=json.load(open('work/p1_rows.json'))
M1=json.load(open('work/fwd_01/meta.json')); M2=json.load(open(FW+'/meta.json'))
n1=M1['n_tokens']; n2=M2['n_tokens']
o1=np.concatenate([[0],np.cumsum(n1)]); o2=np.concatenate([[0],np.cumsum(n2)])
l1=np.concatenate([[0],np.cumsum(np.array(n1)-1)]); l2=np.concatenate([[0],np.cumsum(np.array(n2)-1)])
LP1=np.load('work/fwd_01/token_logprobs.npy'); LP2=np.load(FW+'/token_logprobs.npy')
T1=np.load('work/fwd_01/topk_ids.npy',mmap_mode='r'); T2=np.load(FW+'/topk_ids.npy',mmap_mode='r'); V2=np.load(FW+'/topk_logprobs.npy',mmap_mode='r')
lab_by=collections.defaultdict(dict)
for r in rows: lab_by[r[1]][r[0]-o1[r[1]]]=r[3]
dl=[]; big=[]; forced=collections.defaultdict(list); first_change=[]
for k,ti in enumerate(idx):
    assert n1[ti]==n2[k]
    a=LP1[l1[ti]:l1[ti+1]]; b=LP2[l2[k]:l2[k+1]]; d=b-a   # d[i] = change of logP(token i+1)
    labs=lab_by[ti]
    nat=np.array([not labs.get(i+1,'').startswith('forced') for i in range(len(d))])
    dl.append(np.abs(d[nat]))
    for i in np.nonzero((np.abs(d)>0.3)&nat)[0]:
        big.append((D['meta'][ti]['id'],int(i),labs.get(i,'?'),float(a[i]),float(b[i])))
    for i in range(len(d)):
        L=labs.get(i+1,'')
        if L.startswith('forced'): forced[L].append((a[i],b[i],k,i))
    nz=np.nonzero(np.abs(d[nat])>0.01)[0]; first_change.append(nz[0]/len(d) if len(nz) else 1.0)
dl=np.concatenate(dl)
print('natural tokens %d: mean |dlogp| %.5f, p99 %.4f, >0.1: %d, >0.3: %d'%(len(dl),dl.mean(),np.percentile(dl,99),(dl>0.1).sum(),(dl>0.3).sum()))
print('first |dlogp|>0.01 at fraction of text: median %.2f'%np.median(first_change))
for L in sorted(forced):
    a=np.array([x[0] for x in forced[L]]); b=np.array([x[1] for x in forced[L]])
    print(L,'n %d  P orig median %.3f -> edited median %.3f; edited P>0.5 in %d'%(len(a),np.median(np.exp(a)),np.median(np.exp(b)),(np.exp(b)>0.5).sum()))
# what comes after "Hence answer" and "Hence answer is" in edited
for L,pos_off in [('forced2',0),('forced3',0)]:
    acc=collections.Counter()
    for (_,_,k,i) in forced[L]:
        g=o2[k]+i+1   # row of the forced token itself -> next-token dist
        for t,v in zip(T2[g],V2[g]): acc[int(t)]+=np.exp(v)/len(forced[L])
    print('edited after',L,[(tok.decode([t]).encode('ascii','replace').decode(),round(v,3)) for t,v in acc.most_common(8)])
print('big changes at natural tokens (id, pos, label of position, lp orig, lp edited):')
for x in big[:40]: print(' ',x)
