# Pass-1 analysis: does a corrupted step (F1 arithmetic, F2 mis-copy) change (a) next-paragraph opener probabilities, (b) the residual stream (held-out-by-question AUC per layer)?
import json, sys, numpy as np, collections
from wsp.editor import *
from sklearn.metrics import roc_auc_score
D=sys.argv[1] if len(sys.argv)>1 else 'work/fwd_01'
meta=json.load(open('work/p1_meta.json')); M=json.load(open(D+'/meta.json')); print({k:v for k,v in M['files'].items() if not k.startswith('h') or k in('h24',)}, M.get('positions'))
ids=np.load(D+'/topk_ids.npy'); lp=np.load(D+'/topk_logprobs.npy')
N=len(meta); fam=np.array([m['fam'] for m in meta]); kind=np.array([m['kind'] for m in meta]); qid=np.array([m['id'] for m in meta])
# opener classes by first token string
def cls(s):
    w=s.strip().lower()
    if w in('wait','actually','hmm','hold','oops','no','but','however','check','double','correction','sorry','hang','mistake','error','oh'): return 'recheck'
    if w in('thus','so','therefore','hence','then','now','next'): return 'go'
    return 'other'
strs={int(i):tok.decode([int(i)]) for i in np.unique(ids)}
P=np.zeros((N,3))
for n in range(N):
    for i,l in zip(ids[n],lp[n]): P[n,['recheck','go','other'].index(cls(strs[int(i)]))]+=np.exp(l)
grp=lambda f,k:np.where((fam==f)&(kind==k))[0]
for f,ks in (('F1',('clean','big','small')),('F2',('clean','big'))):
    base=grp(f,'clean')
    for k in ks:
        g=grp(f,k); d=P[g,0]-P[base,0]
        print('%s %-5s n %3d | P(recheck opener) mean %.3f median %.3f | vs clean %+.3f (SE %.3f) | P(go) %.3f | share with recheck top-1 %.2f'%(f,k,len(g),P[g,0].mean(),np.median(P[g,0]),d.mean(),d.std(ddof=1)/np.sqrt(len(d)),P[g,1].mean(),np.mean([cls(strs[int(ids[n,0])])=='recheck' for n in g])))
# top opener tokens after clean vs big
for f in ('F1','F2'):
    for k in ('clean','big'):
        c=collections.Counter()
        for n in grp(f,k):
            for i,l in zip(ids[n][:8],lp[n][:8]): c[strs[int(i)].strip() or repr(strs[int(i)])]+=float(np.exp(l))
        print(f,k,'top openers:',', '.join('%s %.2f'%(w.encode('ascii','replace').decode(),v/len(grp(f,k))) for w,v in c.most_common(12)))
# token log-probs of the result/copied number: clean vs corrupted
try:
    tl=np.load(D+'/token_logprobs.npy',allow_pickle=True); off=np.cumsum([0]+M['n_tokens'])
    print('token_logprobs shape',tl.shape)
    def numlp(n):
        t=meta[n]; text=json.load(open('work/p1_texts.json'))[n] if False else None
    T=json.load(open('work/p1_texts.json'))
    res={}
    for n in range(N):
        seg=tl[off[n]:off[n+1]] if tl.ndim==1 else tl[n]
        if meta[n]['fam']=='F1':
            k=len(tok.encode(meta[n]['cv']+'.\n\n',add_special_tokens=False))   # tokens of result + break (approx: count from the end)
            res[n]=float(np.sum(seg[-k:-1]))
    for k in ('clean','big','small'):
        g=grp('F1',k); v=np.array([res[n] for n in g]); print('F1 %-5s log P(result tokens) mean %.2f median %.2f | P>0.5: %.2f'%(k,v.mean(),np.median(v),(v>np.log(.5)).mean()))
except Exception as e: print('token_logprobs analysis failed',repr(e))
# residual probes: paired difference direction, leave-questions-out
def probe(f,k,name):
    a=grp(f,'clean'); b=grp(f,k); X=np.load(D+'/%s.npy'%name).astype(np.float64)
    qs=np.array(sorted(set(qid[a]))); rng=np.random.RandomState(0); rng.shuffle(qs); folds=np.array_split(qs,5); sc=np.zeros(N); 
    for fo in folds:
        tr=~np.isin(qid[a],fo); te=~tr
        d=(X[b][tr]-X[a][tr]).mean(0); d/=np.linalg.norm(d)
        sc[a[te]]=X[a][te]@d; sc[b[te]]=X[b][te]@d
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; s=np.r_[sc[a],sc[b]]
    pair=(sc[b]>sc[a]).mean()
    dn=np.linalg.norm((X[b]-X[a]).mean(0)); pn=np.linalg.norm(X[b]-X[a],axis=1).mean(); xn=np.linalg.norm(X[a],axis=1).mean()
    return roc_auc_score(y,s),pair,dn,pn,xn
for f,k in (('F1','big'),('F1','small'),('F2','big')):
    print('--',f,k,'(AUC unpaired | paired win share | |mean diff| / mean |pair diff| / |x|)')
    for L in (4,8,12,14,16,18,20,21,22,23,24):
        r=probe(f,k,'h%d'%L); print('  h%-2d AUC %.3f pair %.2f | %.2f / %.2f / %.1f'%((L,)+r))
