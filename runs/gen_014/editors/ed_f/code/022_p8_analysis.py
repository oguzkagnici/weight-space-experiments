# Hypothesis (replication): CUE2 - NON2 (other wording, other control phrases) reproduces the per-question split of CUE - NON from pass 1.
import json, numpy as np, collections
from wsp.editor import *
L='ABCDEFGHIJ'
def load(name,D):
    meta=json.load(open('work/%s_meta.json'%name)); ids=np.load(D+'/topk_ids.npy'); lp=np.load(D+'/topk_logprobs.npy'); R={}
    for n,mt in enumerate(meta):
        if mt['kind']!='ins' or mt['cut']!='late': continue
        d={int(i):float(l) for i,l in zip(ids[n],lp[n])}; fl=lp[n].min()-0.5; v=np.array([d.get(int(t),fl) for t in LETTER_IDS[:10]])
        R[(mt['id'],mt['sample'],mt['arm'])]=(v[L.index(mt['gold'])]-v[L.index(mt['wrong'])],L[int(v.argmax())]==mt['gold'],float(np.exp(v[L.index(mt['gold'])])/np.exp(v).sum()))
    return R
R=load('p1','work/fwd_01'); R.update(load('p8','work/fwd_08'))
qs=sorted(set(k[0] for k in R)); e1=[];e2=[]
print('question      CUE-NON (pass 1)   CUE2-NON2 (pass 8)   [margin log P(gold)-log P(wrong pick), wrong samples]')
for q in qs:
    ss=sorted(set(k[1] for k in R if k[0]==q)); a=np.mean([R[(q,s,'CUE')][0]-R[(q,s,'NON')][0] for s in ss]); c=np.mean([R[(q,s,'CUE2')][0]-R[(q,s,'NON2')][0] for s in ss]); e1.append(a); e2.append(c)
    print('  %s  %+6.2f   %+6.2f'%(q,a,c))
e1=np.array(e1); e2=np.array(e2)
print('correlation of the per-question effects across wordings: r = %.2f (n %d); same sign in %d/%d'%(np.corrcoef(e1,e2)[0,1],len(qs),(np.sign(e1)==np.sign(e2)).sum(),len(qs)))
print('CUE2 - NON2 margin: %+.2f (SE %.2f by question)'%(e2.mean(),e2.std(ddof=1)/np.sqrt(len(e2))))
for a in ('NON','CUE','NON2','CUE2'):
    ks=[k for k in R if k[2]==a]; print('  %-5s top = gold %d/%d, mean P(gold | letters) %.3f | gold by question %s'%(a,sum(R[k][1] for k in ks),len(ks),np.mean([R[k][2] for k in ks]),dict(collections.Counter(k[0][8:] for k in ks if R[k][1]))))
d=np.array([np.mean([float(R[(q,s,'CUE2')][1])-float(R[(q,s,'NON2')][1]) for s in sorted(set(k[1] for k in R if k[0]==q))]) for q in qs]); print('CUE2 - NON2 top = gold: %+.3f (SE %.3f by question)'%(d.mean(),d.std(ddof=1)/np.sqrt(len(d))))
