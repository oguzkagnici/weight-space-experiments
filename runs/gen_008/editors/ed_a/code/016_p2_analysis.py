# Analysis of pass 2: does the unit write "Hence" at sweep ends (dose vs gate z), what follows "Hence", what is the forced
# letter worth at sweep end k=1,2,3 (vs being cut / vs the finished trace's own answer), and are other tokens untouched?
import json, numpy as np, collections, sys
from wsp.editor import *
fwd=sys.argv[1]
P=json.load(open('work/p2_probes.json')); meta=json.load(open(f'{fwd}/meta.json')); off=np.load(f'{fwd}/offsets.npy')
I=np.load(f'{fwd}/topk_ids.npy'); V=np.load(f'{fwd}/topk_logprobs.npy'); TL=np.load(f'{fwd}/token_logprobs.npy')
p1ids=json.load(open('work/p1_ids.json')); off1=np.load('work/fwd_01/offsets.npy'); z1=np.load('work/z_end_p1.npy'); TL1=np.load('work/fwd_01/token_logprobs.npy')
H=tok('Hence')['input_ids'][0]; WE=tok('We')['input_ids'][0]; TH=tok('Thus')['input_ids'][0]
LET=LETTER_IDS
def prob(r,t):
    w=np.where(I[r]==t)[0]; return float(np.exp(V[r][w[0]])) if len(w) else 0.0
suf=6  # tokens of "Hence, the answer is ("
assert len(tok("x.\n\nHence, the answer is (")['input_ids'])-len(tok("x.\n\n")['input_ids'])==suf
rows=[]; after=collections.Counter(); dl=[]; big=0; ntot=0
for i,p in enumerate(P):
    n=meta['n_tokens'][i]; brow=off[i]+n-suf-1; hrow=off[i]+n-suf; last=off[i]+n-1
    ph=prob(brow,H); pw=prob(brow,WE); pt=prob(brow,TH)
    for t,v in zip(I[hrow][:5],V[hrow][:5]): after[tok.decode([int(t)])]+=float(np.exp(v))/len(P)
    lp=np.array([prob(last,t) for t in LET]); forced='ABCDEFGHIJ'[int(lp.argmax())]; conf=lp.max()/max(lp.sum(),1e-9)
    z=np.nan
    if p['id'] in p1ids:
        ti=p1ids.index(p['id']); z=float(z1[off1[ti]+n-suf-1])
        # off-target: logprobs of the prefix tokens (exclude the break row itself) vs pass 1
        a=TL[off[i]-i: off[i]-i+n-suf-2]; b=TL1[off1[ti]-ti: off1[ti]-ti+n-suf-2]
        d=np.abs(a-b); dl.append(d.mean()); big+=int((d>0.1).sum()); ntot+=len(d)
    rows.append(dict(id=p['id'],k=p['k'],cut=p['cut'],correct=p['correct'],gold=p['gold'],pred=p['pred'],z=z,pH=ph,pWe=pw,pThus=pt,forced=forced,conf=float(conf),ok=forced==p['gold'],frac=p['frac']))
json.dump(rows,open('work/p2_rows.json','w'))
r=[x for x in rows if not np.isnan(x['z'])]
zs=np.array([x['z'] for x in r]); ph=np.array([x['pH'] for x in r])
for lo,hi in [(-99,6),(6,8),(8,10),(10,12),(12,14),(14,16),(16,99)]:
    m=(zs>=lo)&(zs<hi); print('z in [%d,%d): n %d  mean P(Hence) %.3f'%(lo,hi,m.sum(),ph[m].mean() if m.sum() else -1))
print('all probes: mean P(Hence) %.3f, frac >0.5: %.2f'%(np.mean([x['pH'] for x in rows]),np.mean([x['pH']>.5 for x in rows])))
print('after Hence (mean top-5 probs):',[(w,round(v,3)) for w,v in after.most_common(12)])
for k in (1,2,3):
    for c in (1,0):
        s=[x for x in rows if x['k']==k and x['cut']==c]
        if not s: continue
        extra='' if c else ' | finished: trace correct %d, forced==pred %d'%(sum(x['correct'] for x in s), sum(x['forced']==x['pred'] for x in s))
        print('k=%d %s n=%d forced correct %d (%.2f) mean conf %.2f%s'%(k,'cut' if c else 'fin',len(s),sum(x['ok'] for x in s),np.mean([x['ok'] for x in s]),np.mean([x['conf'] for x in s]),extra))
print('off-target: mean|dlogp| per token %.5f, tokens with |d|>0.1: %d of %d'%(np.mean(dl),big,ntot))
