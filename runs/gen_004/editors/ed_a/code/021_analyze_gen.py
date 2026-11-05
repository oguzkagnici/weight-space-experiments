# Hypothesis: with the edit, traces commit sooner after their first conclusion (fewer tokens after it, fewer cuts) without changing the answers of finished traces; paired per-question comparison of a generate pass.
import json, re, sys, numpy as np, collections
K=int(sys.argv[1]); d=f'work/fwd_{K:02d}'
O={json.loads(l)['id']:json.loads(l) for l in open(f'{d}/gen_{K}_original.jsonl')}
E={json.loads(l)['id']:json.loads(l) for l in open(f'{d}/gen_{K}_edited.jsonl')}
concl=re.compile(r'(?:^|\n\n)(?:Thus|So|Therefore|Hence)\b[^\n]{0,160}?(?:\b[A-J]\b|\([A-J]\))[^\n]{0,60}(?=\n\n)')
def first_concl(t):
    m=concl.search(t); return m.end()/max(1,len(t)) if m else None
conn=re.compile(r'\b(therefore|Therefore|thus|Thus|hence|Hence)\b')
rows=[]
for q in O:
    o,e=O[q],E[q]
    rows.append((int(o['correct']),int(e['correct']),int(o['cut']),int(e['cut']),int(o['tokens']),int(e['tokens'])))
    fo,fe=first_concl(o['think']),first_concl(e['think'])
    rep=len(re.findall(r'\b(\w+)( \1\b){2,}',e['raw']))
    print(f"{q[8:]:6s} {o['category'][:6]:6s} gold {o['gold']} orig {o['pred'] or '-'}{'*' if int(o['correct']) else ' '} cut{o['cut']} {o['tokens']:>4} | edit {e['pred'] or '-'}{'*' if int(e['correct']) else ' '} cut{e['cut']} {e['tokens']:>4} | conn/1k {1000*len(conn.findall(o['think']))/max(1,int(o['tokens'])):.1f}->{1000*len(conn.findall(e['think']))/max(1,int(e['tokens'])):.1f} rep{rep}")
r=np.array(rows)
n=len(r); dacc=r[:,1]-r[:,0]
print('n',n,'acc orig %.3f edit %.3f diff %+.3f (SE %.3f) w->r %d r->w %d'%(r[:,0].mean(),r[:,1].mean(),dacc.mean(),dacc.std(ddof=1)/np.sqrt(n),((r[:,0]==0)&(r[:,1]==1)).sum(),((r[:,0]==1)&(r[:,1]==0)).sum()))
print('cut orig %.3f edit %.3f; tokens orig %.0f edit %.0f'%(r[:,2].mean(),r[:,3].mean(),r[:,4].mean(),r[:,5].mean()))
fin=r[:,2]==0
print('orig-finished subset n %d: acc %.3f -> %.3f, tokens %.0f -> %.0f'%(fin.sum(),r[fin,0].mean(),r[fin,1].mean(),r[fin,4].mean(),r[fin,5].mean()))
