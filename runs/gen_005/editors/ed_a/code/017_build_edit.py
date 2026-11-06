# Hypothesis: a repurposed quiet L23 neuron (n74) gated on the churn-break direction fires on ~half of churn breaks, on almost no ordinary tokens and on few finished-trace breaks; its output writes Hence/Therefore up and churn openers (Alternatively/Maybe/Perhaps/Another/Possibly) down.
import json, sys, numpy as np, torch
from wsp.editor import *
KAPPA=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
PCT=float(sys.argv[2]) if len(sys.argv)>2 else 40
SAVE=len(sys.argv)>3 and sys.argv[3]=='save'
L=23; N=74
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json')); rowsI=np.array([r['row'] for r in recs])
cls=np.array([r['cls'] for r in recs]); cut=np.array([r['cut'] for r in recs]); ti=np.array([r['ti'] for r in recs])
X=np.load(f'{D}/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
Xb=torch.tensor(np.asarray(X[rowsI],np.float32))
offs=np.load(f'{D}/offsets.npy'); tok_ti=np.repeat(np.arange(len(offs)-1),np.diff(offs))
isbrk=np.zeros(X.shape[0],bool); isbrk[rowsI]=True
rng=np.random.default_rng(3); samp=np.sort(rng.choice(np.where(~isbrk)[0],60000,replace=False))
Xs=torch.tensor(np.asarray(X[samp],np.float32))
mu=Xs.mean(0); mh=mu/mu.norm(); mp=float((Xs@mh).mean())
ch=Xb[cls=='churn'].mean(0)
g1=ch-Xb[cls=='other'].mean(0); g2=ch-Xs.mean(0)
d=g1/g1.norm()+g2/g2.norm(); d=d/d.norm()
sb=Xb@d
theta=float(np.percentile(sb[cls=='churn'].numpy(),PCT))
w1=(d-(theta/mp)*mh)            # w1.x = s - theta*p/mp
w3=mh/mp                         # w3.x ~ 1
def act(Xm): return torch.nn.functional.silu(Xm@w1)*(Xm@w3)
ab=act(Xb).numpy(); as_=act(Xs).numpy()
print(f'theta {theta:.2f} (pct {PCT}); act at churn breaks: fire(>0.1) {(ab[cls=="churn"]>0.1).mean():.2f} median-if-fire {np.median(ab[(cls=="churn")&(ab>0.1)]):.2f}')
for c in (0,1):
    for k in ('churn','other','commit'):
        m=(cls==k)&(cut==c)
        if m.sum(): print(f'  cut={c} {k:6s} n {m.sum():4d} fire {(ab[m]>0.1).mean():.3f} mean act {ab[m].mean():+.3f}')
print(f'  non-break tokens: fire {(as_>0.1).mean()*100:.3f}%  min act {as_.min():.4f} max {as_.max():.2f}')
# fires per trace
fires=np.zeros(len(offs)-1); 
for i in np.where(ab>0.1)[0]: fires[ti[i]]+=1
cutT=np.array([1 if c else 0 for c in [None]*0])
ids_list=json.load(open('work/req1_ids.json'))
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
cT=np.array([rows[q]['cut'] for q in ids_list])
print('  fires per trace: cut mean %.2f (traces with >=1: %d/%d), finished mean %.2f (%d/%d)'%(fires[cT==1].mean(),(fires[cT==1]>0).sum(),(cT==1).sum(),fires[cT==0].mean(),(fires[cT==0]>0).sum(),(cT==0).sum()))
# first firing position (fraction) in finished traces
for i in np.where(cT==0)[0]:
    if fires[i]>0:
        fr=[recs[j]['frac'] for j in np.where((ab>0.1)&(ti==i))[0]]
        print('   finished trace',ids_list[i],rows[ids_list[i]]['correct'],'fires at',[round(f,2) for f in fr])
# output column
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight'); gp=load(f'model.layers.{L}.post_ffn_norm.weight')
up=[tok(s,add_special_tokens=False).input_ids[0] for s in ['Hence','Therefore']]
dn=[tok(s,add_special_tokens=False).input_ids[0] for s in ['Alternatively','Maybe','Perhaps','Another','Possibly']]
t=gf*(E[up].mean(0)-E[dn].mean(0))
u=gp*t; u=u/u.norm()
# direct lens of the written direction (post-norm g_p*u then final-norm gain)
lo=E@(gf*(gp*u)); z=(lo-lo.mean())/lo.std()
print('  output lens z:', ' '.join(f'{s}:{float(z[tok(s,add_special_tokens=False).input_ids[0]]):+.1f}' for s in ['Hence','Therefore','Thus','So','Alternatively','Maybe','Perhaps','Another','Let','But','Given','I','The',' the',' answer','A','B']))
print('  top tokens of output lens (ids):', torch.topk(lo,8).indices.tolist(), [tok.decode([i]) for i in torch.topk(lo,8).indices.tolist()])
if SAVE:
    W1=load(f'model.layers.{L}.feed_forward.w1.weight').clone(); W3=load(f'model.layers.{L}.feed_forward.w3.weight').clone(); W2=load(f'model.layers.{L}.feed_forward.w2.weight').clone()
    W1[N]=w1; W3[N]=w3; W2[:,N]=KAPPA*u
    print(save_edit(f'model.layers.{L}.feed_forward.w1.weight',W1)); print(save_edit(f'model.layers.{L}.feed_forward.w3.weight',W3)); print(save_edit(f'model.layers.{L}.feed_forward.w2.weight',W2))
    np.save('work/017_gate_d.npy',d.numpy()); json.dump({'theta':theta,'mp':mp,'kappa':KAPPA,'pct':PCT},open('work/017_params.json','w'))
