# Hypothesis: with edit v1, P(Hence/Therefore) rises and P(churn openers) falls at churn breaks in proportion to the neuron's activation, while token log-probs elsewhere are unchanged.
import json, sys, numpy as np, torch
from wsp.editor import *
D1='work/fwd_01'; D2=sys.argv[1] if len(sys.argv)>1 else 'work/fwd_02'
recs=json.load(open('work/009_breaks.json')); rowsI=np.array([r['row'] for r in recs])
cls=np.array([r['cls'] for r in recs]); cut=np.array([r['cut'] for r in recs])
p=json.load(open('work/017_params.json'))
X=np.load(f'{D1}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); Xb=torch.tensor(np.asarray(X[rowsI],np.float32))
W1=load('model.layers.23.feed_forward.w1.weight'); W3=load('model.layers.23.feed_forward.w3.weight')
a=(torch.nn.functional.silu(Xb@W1[74])*(Xb@W3[74])).numpy()
up=[tok(s,add_special_tokens=False).input_ids[0] for s in ['Hence','Therefore']]
th=[tok(s,add_special_tokens=False).input_ids[0] for s in ['Thus']]
dn=[tok(s,add_special_tokens=False).input_ids[0] for s in ['Alternatively','Maybe','Perhaps','Another','Possibly']]
def mass(D,ids):
    I=np.load(f'{D}/topk_ids.npy',mmap_mode='r')[rowsI]; V=np.exp(np.load(f'{D}/topk_logprobs.npy',mmap_mode='r')[rowsI])
    return (V*np.isin(I,ids)).sum(1)
pu1,pu2=mass(D1,up),mass(D2,up); pt1,pt2=mass(D1,th),mass(D2,th); pd1,pd2=mass(D1,dn),mass(D2,dn)
for c in (0,1):
  for k in ('churn','other','commit'):
    for fire in (True,False):
        m=(cls==k)&(cut==c)&((a>0.1)==fire)
        if m.sum()<3: continue
        print(f'cut={c} {k:6s} fire={int(fire)} n {m.sum():4d} act {a[m].mean():.2f} | P(Hence/Therefore) {pu1[m].mean():.3f}->{pu2[m].mean():.3f} | P(Thus) {pt1[m].mean():.3f}->{pt2[m].mean():.3f} | P(churn) {pd1[m].mean():.3f}->{pd2[m].mean():.3f}')
# dose response by activation bins (all breaks)
for lo,hi in ((0.1,0.5),(0.5,1),(1,2),(2,4),(4,99)):
    m=(a>=lo)&(a<hi)
    if m.sum(): print(f'act {lo}-{hi}: n {m.sum():3d} P(H/T) {pu1[m].mean():.3f}->{pu2[m].mean():.3f}  P(churn) {pd1[m].mean():.3f}->{pd2[m].mean():.3f}')
# token logprob changes elsewhere
l1=np.load(f'{D1}/token_logprobs.npy'); l2=np.load(f'{D2}/token_logprobs.npy')
d=l2-l1
# map rows: token_logprobs has n-1 entries per text; offsets
offs=np.load(f'{D1}/offsets.npy'); n=np.diff(offs)
lpo=np.concatenate([[0],np.cumsum(n-1)])
brk_lp=np.concatenate([[lpo[r['ti']]+r['t']] for r in recs])
mask=np.ones(len(d),bool); mask[brk_lp]=False
print('non-break positions: mean|dlogp| %.5f max %.3f, n>0.1: %d of %d'%(np.abs(d[mask]).mean(),np.abs(d[mask]).max(),(np.abs(d[mask])>0.1).sum(),mask.sum()))
print('break positions (actual next token): mean dlogp %.4f, n with |d|>0.1: %d'%(d[~mask].mean(),(np.abs(d[~mask])>0.1).sum()))
np.save('work/019_act.npy',a)
