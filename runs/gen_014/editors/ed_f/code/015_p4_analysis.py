# Hypothesis (pass 4): on never-right cue questions the model prefers the premise of its wrong pick over the right implication (false belief), bare and inside its trace; on mixed/responsive questions it prefers the right implication. And a cue quote does not disturb right conclusions.
import json, numpy as np, collections
from wsp.editor import *
meta=json.load(open('work/p4_meta.json')); ids=np.load('work/fwd_04/topk_ids.npy'); lp=np.load('work/fwd_04/topk_logprobs.npy'); L='ABCDEFGHIJ'
t1,t2=tok.encode('1',add_special_tokens=False)[0],tok.encode('2',add_special_tokens=False)[0]
R=collections.defaultdict(list)
for n,mt in enumerate(meta):
    d={int(i):float(l) for i,l in zip(ids[n],lp[n])}; fl=lp[n].min()-0.5
    if mt['kind']=='know':
        a,b=(LETTER_IDS[0],LETTER_IDS[1]) if mt['ctx']=='bare' else (t1,t2)
        la,lb=d.get(int(a),fl),d.get(int(b),fl); m=(la-lb) if mt['order']==0 else (lb-la)       # log-odds for the RIGHT statement
        R[(mt['id'],mt['ctx'],mt.get('status','-'),mt.get('sample',-1))].append((m,np.exp(la)+np.exp(lb)))
agg={}
for k,v in R.items(): agg[k]=(np.mean([x[0] for x in v]),np.mean([x[1] for x in v]))
print('log-odds (right implication over wrong premise), mean of the two orders; bare | in-trace by sample status')
resp={'mmlupro_2654','mmlupro_6110','mmlupro_6614','mmlupro_6800'}
rows=[]
for q in sorted(set(k[0] for k in agg)):
    bare=agg[(q,'bare','-',-1)][0]; tr={s:[agg[k][0] for k in agg if k[0]==q and k[1]=='trace' and k[2]==s] for s in ('wrong','right','cut')}
    rows.append((q,bare,tr)); f=lambda v:('%+.2f (n %d)'%(np.mean(v),len(v))) if v else '   --    '
    print('  %s %s bare %+.2f | trace: wrong %s right %s cut %s'%(q,'CUE-RESPONSIVE' if q in resp else '              ',bare,f(tr['wrong']),f(tr['right']),f(tr['cut'])))
b=np.array([r[1] for r in rows]); print('bare: prefers the right implication in %d of %d questions (mean log-odds %+.2f); responsive questions %d/4, others %d/9'%((b>0).sum(),len(b),b.mean(),sum(r[1]>0 for r in rows if r[0] in resp),sum(r[1]>0 for r in rows if r[0] not in resp)))
tw=[(r[0],np.mean(r[2]['wrong'])) for r in rows if r[2]['wrong']]
print('in-trace (wrong samples, question means): prefers the right implication in %d of %d questions; responsive %d/4, others %d/%d'%(sum(v>0 for q,v in tw),len(tw),sum(v>0 for q,v in tw if q in resp),sum(v>0 for q,v in tw if q not in resp),sum(1 for q,v in tw if q not in resp)))
print('mean probability mass on the two answer tokens: bare %.2f, trace %.2f'%(np.mean([v[1] for k,v in agg.items() if k[1]=='bare']),np.mean([v[1] for k,v in agg.items() if k[1]=='trace'])))
# safety: right samples, late cut
print('\nright samples of the cue questions, late cut: top letter / P(gold | letters) by arm')
S=collections.defaultdict(dict)
for n,mt in enumerate(meta):
    if mt['kind']!='ins': continue
    d={int(i):float(l) for i,l in zip(ids[n],lp[n])}; v=np.array([d.get(int(t),lp[n].min()-0.5) for t in LETTER_IDS[:10]]); p=np.exp(v)/np.exp(v).sum()
    S[(mt['id'],mt['sample'])][mt['arm']]=(L[int(v.argmax())],p[L.index(mt['gold'])],mt['gold'])
for k,v in sorted(S.items()): print('  ',k[0],k[1],'gold',v['BASE'][2],' '.join('%s: %s %.2f'%(a,v[a][0],v[a][1]) for a in ('BASE','NON','CUE')))
for a in ('BASE','NON','CUE'): print('  %s: top = gold %d/%d, mean P(gold) %.3f'%(a,sum(v[a][0]==v[a][2] for v in S.values()),len(S),np.mean([v[a][1] for v in S.values()])))
