# Hypothesis: (paired check) compare an edited generate pass with the parent's cached samples (passk bank or holdout48 bank) by id and sample: accuracy, both-finish diff, first divergence.
import json, sys, glob, numpy as np, collections
f=glob.glob(sys.argv[1]+'/gen_*_edited.jsonl')[0]; bank=sys.argv[2]
E=[json.loads(l) for l in open(f)]
B={(r['id'],r.get('sample',0)):r for r in map(json.loads,open(bank))}
d=[];both=[];div=[];ident=0;rows=[]
for e in E:
    k=(e['id'],e.get('sample',0)); p=B[k]
    d.append(e['correct']-p['correct'])
    if not e['cut'] and not p['cut']: both.append(e['correct']-p['correct'])
    a,b=p['raw'],e['raw']; i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    if a==b: ident+=1
    div.append(i/max(len(a),1))
    rows.append((k, p['pred'],p['correct'],p['cut'], e['pred'],e['correct'],e['cut'], round(i/max(len(a),1),2)))
d=np.array(d); both=np.array(both)
se=lambda x: x.std(ddof=1)/np.sqrt(len(x))*100 if len(x)>1 else float('nan')
print('n',len(d),'parent acc %.1f edited acc %.1f diff %+.1f (SE %.1f) +%d/-%d'%(np.mean([B[(e['id'],e.get('sample',0))]['correct'] for e in E])*100,np.mean([e['correct'] for e in E])*100,d.mean()*100,se(d),(d>0).sum(),(d<0).sum()))
print('both-finish n',len(both),'diff %+.1f (SE %.1f) +%d/-%d'%(both.mean()*100,se(both),(both>0).sum(),(both<0).sum()))
print('cut parent',sum(B[(e['id'],e.get('sample',0))]['cut'] for e in E),'edited',sum(e['cut'] for e in E),'identical',ident,'median first-diff frac %.2f'%np.median(div))
for r in rows: print(r)
