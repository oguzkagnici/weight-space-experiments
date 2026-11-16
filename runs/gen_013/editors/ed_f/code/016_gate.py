# GATE (pass 3 vs pass 1): does the edited model raise re-check openers after corrupted steps more than after clean ones on HELD-OUT questions, and how much does it change other tokens?
import json, sys, numpy as np
from wsp.editor import *
D0='work/fwd_01'; D1=sys.argv[1]; meta=json.load(open('work/p1_meta.json')); trq=set(json.load(open('work/train_questions.json'))); N=len(meta)
def cls(s):
    w=s.strip().lower()
    return w in('wait','actually','hmm','hold','oops','no','but','however','check','double','correction','sorry','hang','mistake','error','oh')
def prec(D):
    ids=np.load(D+'/topk_ids.npy'); lp=np.load(D+'/topk_logprobs.npy'); st={int(i):cls(tok.decode([int(i)])) for i in np.unique(ids)}
    return np.array([sum(np.exp(l) for i,l in zip(ids[n],lp[n]) if st[int(i)]) for n in range(N)]), ids[:,0]
P0,t0=prec(D0); P1,t1=prec(D1)
fam=np.array([m['fam'] for m in meta]); kind=np.array([m['kind'] for m in meta]); te=np.array([m['id'] not in trq for m in meta])
for split,msk in (('HELD-OUT questions',te),('fit questions',~te)):
    print(split)
    for f,ks in (('F1',('clean','big','small')),('F2',('clean','big'))):
        base=np.where((fam==f)&(kind=='clean')&msk)[0]
        for k in ks:
            g=np.where((fam==f)&(kind==k)&msk)[0]; d=P1[g]-P0[g]
            gap0=(P0[g]-P0[base]).mean(); gap1=(P1[g]-P1[base]).mean(); gd=(P1[g]-P1[base])-(P0[g]-P0[base])
            print('  %s %-5s n %2d | P(re-check opener) parent %.3f -> edited %.3f (%+.3f, SE %.3f) | corrupted-minus-clean gap %.3f -> %.3f (change %+.3f, SE %.3f) | top-1 changed %d'%(f,k,len(g),P0[g].mean(),P1[g].mean(),d.mean(),d.std(ddof=1)/np.sqrt(len(g)),gap0,gap1,gd.mean(),gd.std(ddof=1)/np.sqrt(len(g)),(t0[g]!=t1[g]).sum()))
a=np.load(D0+'/token_logprobs.npy'); b=np.load(D1+'/token_logprobs.npy'); d=np.abs(a-b)
print('off-target over %d teacher-forced tokens: mean |dlogp| %.5f, median %.6f, share > 0.01: %.4f, > 0.1: %.4f, max %.2f'%(len(d),d.mean(),np.median(d),(d>.01).mean(),(d>.1).mean(),d.max()))
