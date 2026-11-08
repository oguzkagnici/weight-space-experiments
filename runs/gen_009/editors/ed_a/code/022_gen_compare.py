# Hypothesis: paired against the bank (byte-identical parent), the edit converts cut traces with eval sweeps into finished answers stated at a sweep end, with few losses.
import json, sys, glob, numpy as np, re
bankf=sys.argv[2] if len(sys.argv)>2 else '/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl'
B={json.loads(l)['id']:json.loads(l) for l in open(bankf)}
E=[json.loads(l) for f in sys.argv[1].split(',') for l in open(glob.glob(f+'/gen_*_edited.jsonl')[0])]
d=[];res=[]
for e in E:
    b=B[e['id']]
    fd=next((i for i,(x,y) in enumerate(zip(b['raw'],e['raw'])) if x!=y),min(len(b['raw']),len(e['raw'])))
    ins=len(re.findall(r'\n\nHence answer',e['think'])); insb=len(re.findall(r'\n\nHence answer',b['think']))
    d.append(e['correct']-b['correct'])
    res.append((e['id'],e['category'],b['correct'],e['correct'],b['cut'],e['cut'],b['tokens'],e['tokens'],fd/max(1,len(b['raw'])),ins,insb))
d=np.array(d); n=len(d)
print('n %d parent %.1f edited %.1f diff %+.1f SE %.1f  +%d/-%d  cut %d->%d  tokens %.0f->%.0f  identical %d'%(n,100*np.mean([r[2] for r in res]),100*np.mean([r[3] for r in res]),100*d.mean(),100*d.std(ddof=1)/np.sqrt(n),(d>0).sum(),(d<0).sum(),sum(r[4] for r in res),sum(r[5] for r in res),np.mean([r[6] for r in res]),np.mean([r[7] for r in res]),sum(r[8]>=1 for r in res)))
print('first diff median frac %.2f'%np.median([r[8] for r in res]))
fired=[r for r in res if r[9]>0]
print('traces with inserted "Hence answer": %d; among them parent->edited correct %d->%d, cut %d->%d'%(len(fired),sum(r[2] for r in fired),sum(r[3] for r in fired),sum(r[4] for r in fired),sum(r[5] for r in fired)))
nf=[r for r in res if r[9]==0]
print('without: %d; correct %d->%d'%(len(nf),sum(r[2] for r in nf),sum(r[3] for r in nf)))
for r in res:
    if r[2]!=r[3] or r[9]: print('  %s %-12s corr %d->%d cut %d->%d tok %d->%d firstdiff %.2f HenceAns %d'%r[:10])
