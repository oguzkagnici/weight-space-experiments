# Hypothesis (step 1b): at option-judging positions, right samples put more attention on the stem than wrong samples of the same question (per layer and head).
import json, sys, numpy as np, collections
name='p3b'; M=np.load('work/p3b_attn_regions.npy'); meta=json.load(open('work/p3b_meta.json')); REG=json.load(open('work/attn_regions_meta.json'))['REG']; LAY=[2,4,6,8,10,12,14,16,18,20,22]
S=M[...,REG.index('stem')]+M[...,REG.index('cue_src')]            # texts x layers x heads
O=M[...,REG.index('options')]; T=M[...,REG.index('trace')]
J=[i for i,m in enumerate(meta) if m['kind']=='judge']
kind=np.array([m['kind'] for m in meta])
print('mean attention by region, all layers/heads: judging positions',{r:round(float(M[J][...,k].mean()),3) for k,r in enumerate(REG)})
print('  background trace positions',{r:round(float(M[kind=='bg'][...,k].mean()),3) for k,r in enumerate(REG)})
FULL=[0,2,4,6,8,10]
print('stem attention per full-attention layer at judging positions:',{LAY[l]:round(float(S[J][:,l].mean()),3) for l in FULL},'| at background positions:',{LAY[l]:round(float(S[kind=='bg'][:,l].mean()),3) for l in FULL})
def within(X,tgt):
    by=collections.defaultdict(lambda:{'right':[],'wrong':[]})
    for i in J:
        m=meta[i]
        if tgt is None or m['tgt']==tgt: by[m['id']][m['status']].append(X[i])
    d=np.array([np.mean(v['right'],0)-np.mean(v['wrong'],0) for v in by.values() if v['right'] and v['wrong']])
    r=np.mean([np.mean(v['right'],0) for v in by.values() if v['right'] and v['wrong']],0); w=np.mean([np.mean(v['wrong'],0) for v in by.values() if v['right'] and v['wrong']],0)
    return d,r,w
for nm,X in (('STEM',S),('OPTIONS',O),('TRACE',T)):
    for tgt in (None,'gold','wrong'):
        d,r,w=within(X[:,FULL].mean((1,2)),tgt)
        print('%s attention (mean of full-attention layers), judging %s option: right %.4f wrong %.4f | within-question diff %+.4f (SE %.4f, n %d questions, right>wrong %d)'%(nm,tgt or 'gold+wrong',r,w,d.mean(),d.std(ddof=1)/np.sqrt(len(d)),len(d),(d>0).sum()))
d,r,w=within(S,None); n=len(d); z=d.mean(0)/(d.std(0,ddof=1)/np.sqrt(n)+1e-9)
print('per layer (all 16 heads): within-question right-wrong diff of stem attention:',{LAY[l]:'%+.4f (z %+.1f)'%(d[:,l].mean(),d[:,l].mean(1).mean()/(d[:,l].mean(1).std(ddof=1)/np.sqrt(n)+1e-9)) for l in range(11)})
hs=[(z[l,h],LAY[l],h,r[l,h],w[l,h]) for l in FULL for h in range(16) if max(r[l,h],w[l,h])>0.02]
print('heads with stem attention > .02 (%d): |z| > 2 in %d (expected by chance %.1f)'%(len(hs),sum(abs(x[0])>2 for x in hs),0.05*len(hs)))
for zz,l,h,rr,ww in sorted(hs,key=lambda x:-abs(x[0]))[:10]: print('   L%d.H%d right %.3f wrong %.3f z %+.1f'%(l,h,rr,ww,zz))
top=sorted([(S[J][:,l,h].mean(),LAY[l],h) for l in range(11) for h in range(16)],reverse=True)[:12]
print('heads reading the stem most at judging positions:',[(l,h,round(float(v),3)) for v,l,h in top])
