# Paired analysis of "both" generate passes (fresh holdout): accuracy diff, swaps, cut transitions, first differing char, inserted Hence paragraphs.
import json, sys, re, numpy as np
O={};E={}
for d in sys.argv[1:]:
    import glob
    for l in open(glob.glob(d+'/gen_*_original.jsonl')[0]): r=json.loads(l); O[r['id']]=r
    for l in open(glob.glob(d+'/gen_*_edited.jsonl')[0]): r=json.loads(l); E[r['id']]=r
ids=sorted(O); d=np.array([E[i]['correct']-O[i]['correct'] for i in ids]); n=len(d)
print('n %d parent %.1f edited %.1f diff %+.1f (SE %.1f) +%d/-%d; cut %d -> %d; identical %d'%(n,100*np.mean([O[i]['correct'] for i in ids]),100*np.mean([E[i]['correct'] for i in ids]),100*d.mean(),100*d.std(ddof=1)/np.sqrt(n),(d>0).sum(),(d<0).sum(),sum(O[i]['cut'] for i in ids),sum(E[i]['cut'] for i in ids),sum(O[i]['raw']==E[i]['raw'] for i in ids)))
fd=[]
for i in ids:
    a,b=O[i]['raw'],E[i]['raw']; k=next((j for j in range(min(len(a),len(b))) if a[j]!=b[j]),min(len(a),len(b))); fd.append(k/max(1,len(a)))
    nh=len(re.findall(r'\n\nHence',E[i]['think']))-len(re.findall(r'\n\nHence',O[i]['think']))
    if O[i]['correct']!=E[i]['correct'] or O[i]['cut']!=E[i]['cut']:
        print(' ',i,O[i]['category'][:6],'cor %d->%d cut %d->%d tok %d->%d firstdiff %.2f extraHence %d'%(O[i]['correct'],E[i]['correct'],O[i]['cut'],E[i]['cut'],O[i]['tokens'],E[i]['tokens'],k/max(1,len(a)),nh))
print('median first diff %.2f'%np.median(fd))
