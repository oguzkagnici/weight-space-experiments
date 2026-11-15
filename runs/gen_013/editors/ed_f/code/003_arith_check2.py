# Hypothesis: genuine arithmetic slips (binary a op b = c, c complete, wrong beyond rounding) are frequent and associate with wrong/cut traces within question.
import json, collections, re, math, sys
import numpy as np
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
def load_rows(extra=()):
    rows=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
    seen={(r['id'],r['sample']) for r in rows}
    for l in open(S+'trace_bank_gen008champ_cap8192_det1.jsonl'):
        r=json.loads(l); r.setdefault('sample',0)
        if (r['id'],r['sample']) not in seen: rows.append(r)
    return rows
NUM=r'\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?'
pat=re.compile(r'(?<![\w.,^/*+\-×·(])('+NUM+r')\s*(\+|-|\*|×|/|÷|\\times|\\cdot|·)\s*('+NUM+r')\s*(=|≈|\\approx)\s*(-?'+NUM+r')')
def f(s): return float(s.replace(',',''))
def ndec(s):
    s=s.replace(',','').lower().split('e')[0]
    return len(s.split('.')[1]) if '.' in s else 0
def check(a,op,b,eq,c):
    A,B,C=f(a),f(b),f(c)
    if op=='+': v=A+B
    elif op=='-': v=A-B
    elif op in('/','÷'):
        if B==0: return None
        v=A/B
    else: v=A*B
    d=ndec(c)
    if 'e' in c.lower():
        mant=f(c.lower().split('e')[0]); ex=int(c.lower().split('e')[1]); unit=10**(ex-d)
    else: unit=10**(-d)
    err=abs(C-v)
    # ok if within one unit of the last written place (rounding or truncation); '=' or approx alike
    tol=1.01*unit if (eq!='=' or d>0 or not float(v).is_integer()) else 0.0
    if not float(v).is_integer() and d==0: tol=1.01
    exact_int = float(v).is_integer() and abs(v)<1e15
    return v, err<=tol+1e-12*abs(v), err/abs(v) if v else (0 if err==0 else 9), exact_int
def scan(rows):
    out=[]
    for r in rows:
        t=r['raw']
        for m in pat.finditer(t):
            a,op,b,eq,c=m.groups()
            nxt=t[m.end():m.end()+12]
            if re.match(r'\d|,\d|\.\d|\s*(\*|×|/|÷|\^|\\times|\\cdot|·|x\s*10|e\d|%)',nxt): continue   # result continues / is part of a longer expression
            if re.match(r'\s*[\+\-]\s*\d',nxt): continue
            pre=t[max(0,m.start()-12):m.start()]
            if re.search(r'(times|cdot|frac\{?|sqrt\{?)\s*$',pre): continue
            try: res=check(a,op,b,eq,c)
            except Exception: continue
            if res is None: continue
            v,ok,rel,exi=res
            out.append(dict(id=r['id'],sample=r['sample'],cat=r['category'],correct=r['correct'],cut=r['cut'],start=m.start(),cstart=m.start(5),cend=m.end(5),a=a,op=op,b=b,eq=eq,c=c,true=v,ok=ok,rel=rel,exact_int=exi,expr=m.group(0)))
    return out
if __name__=='__main__':
    rows=load_rows(); eqs=scan(rows)
    bad=[e for e in eqs if not e['ok']]
    print('traces',len(rows),'equations',len(eqs),'bad',len(bad),'(%.1f%%)'%(100*len(bad)/len(eqs)))
    opn=lambda o:{'\\times':'*','×':'*','\\cdot':'*','·':'*','÷':'/'}.get(o,o)
    c1=collections.Counter(opn(e['op']) for e in eqs); c2=collections.Counter(opn(e['op']) for e in bad)
    print('by op: '+', '.join('%s %d/%d (%.1f%%)'%(o,c2[o],c1[o],100*c2[o]/c1[o]) for o in c1))
    rels=np.array([e['rel'] for e in bad]); print('bad rel err quantiles',np.round(np.quantile(rels,[.1,.25,.5,.75,.9]),4),'share rel<2%%: %.2f, >50%%: %.2f'%((rels<.02).mean(),(rels>.5).mean()))
    # first occurrence vs repeats
    seen=set(); first=[]
    for e in bad:
        k=(e['id'],e['sample'],e['a'],opn(e['op']),e['b'],e['c'])
        if k not in seen: seen.add(k); first.append(e)
    print('distinct bad statements',len(first))
    # per trace
    per=collections.defaultdict(lambda:[0,0]); info={}
    for r in rows: info[(r['id'],r['sample'])]=r; per[(r['id'],r['sample'])]
    for e in eqs:
        per[(e['id'],e['sample'])][0]+=1; per[(e['id'],e['sample'])][1]+= (not e['ok'])
    for tag,sel in [('right',lambda r:r['correct']),('wrong finished',lambda r:not r['correct'] and not r['cut']),('cut',lambda r:r['cut'])]:
        ks=[k for k in per if sel(info[k]) and per[k][0]>=3]
        print('%-15s traces with >=3 eqs: %3d | eqs/trace %.1f | bad share %.3f | traces with >=1 bad %.2f'%(tag,len(ks),np.mean([per[k][0] for k in ks]),sum(per[k][1] for k in ks)/sum(per[k][0] for k in ks),np.mean([per[k][1]>0 for k in ks])))
    json.dump(eqs,open('work/eqs.json','w'))
    n=int(sys.argv[1]) if len(sys.argv)>1 else 50
    import random; random.seed(0); random.shuffle(first)
    for e in first[:n]:
        print(e['id'],e['sample'],e['cat'][:5],'ok' if e['correct'] else ('cut' if e['cut'] else 'WRONG'),'|',e['expr'].encode('ascii','replace').decode(),'| true %.8g rel %.4f'%(e['true'],e['rel']))
