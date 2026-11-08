# Paired analysis of an edited-only generate pass on bank questions vs the bank (byte-identical reference for the parent):
# accuracy change, wrong->right / right->wrong, cut transitions, first differing char, and whether the change starts at a "Hence" commit.
import json, sys, glob, numpy as np, re
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
files=sys.argv[1:]
E=[json.loads(l) for f in files for l in open(f)]
d=[];rows=[]
for e in E:
    b=B[e['id']]; a,bb=b['raw'],e['raw']
    k=next((i for i in range(min(len(a),len(bb))) if a[i]!=bb[i]), min(len(a),len(bb)))
    ident=a==bb
    ctx=bb[max(0,k-1):k+12].replace('\n','/')
    hence_at_div=bb[k:k+6].startswith('Hence') or bb[max(0,k-2):k+6].find('Hence')>=0
    nh=len(re.findall(r'\n\nHence',e['think']))-len(re.findall(r'\n\nHence',b['think']))
    rows.append((e['id'],b['category'][:6],b['correct'],e['correct'],b['cut'],e['cut'],b['tokens'],e['tokens'],ident,round(k/max(1,len(a)),2),hence_at_div,nh,ctx))
    d.append(e['correct']-b['correct'])
d=np.array(d); n=len(d)
print('n %d  parent acc %.1f  edited acc %.1f  diff %+.1f (SE %.1f)  +%d/-%d'%(n,100*np.mean([r[2] for r in rows]),100*np.mean([r[3] for r in rows]),100*d.mean(),100*d.std(ddof=1)/np.sqrt(n),(d>0).sum(),(d<0).sum()))
print('cut %d -> %d; byte-identical %d; median first-diff frac %.2f; divergence starts at a Hence %d'%(sum(r[4] for r in rows),sum(r[5] for r in rows),sum(r[8] for r in rows),np.median([r[9] for r in rows if not r[8]] or [1]),sum(r[10] for r in rows)))
print('tokens parent %.0f edited %.0f'%(np.mean([r[6] for r in rows]),np.mean([r[7] for r in rows])))
for r in rows:
    if r[2]!=r[3] or r[4]!=r[5]: print(' ',r[0],r[1],'cor %d->%d cut %d->%d tok %d->%d first-diff %.2f hence@div %s extraHence %d %r'%(r[2],r[3],r[4],r[5],r[6],r[7],r[9],r[10],r[11],r[12].encode('ascii','replace').decode()))
