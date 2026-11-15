# Hypothesis: arithmetic slips (a op b = c with wrong c) are a frequent, automatically detectable step error in parent traces.
import json, collections, re, math, sys
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
rows=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
seen={(r['id'],r['sample']) for r in rows}
for l in open(S+'trace_bank_gen008champ_cap8192_det1.jsonl'):
    r=json.loads(l); r.setdefault('sample',0)
    if (r['id'],r['sample']) not in seen: rows.append(r)
print('traces',len(rows))
num=r'(?<![\w.])(\d[\d,]*\.?\d*(?:e[+-]?\d+)?)'
pat=re.compile(num+r'\s*(\+|-|\*|×|x|/|÷|\\times|\\cdot|·)\s*'+num.replace('(?<![\\w.])','')+r'\s*(=|≈|\\approx|~)\s*(-?\d[\d,]*\.?\d*(?:e[+-]?\d+)?)(?![\d]*\s*[\^*×/+x·\\])')
def f(s):
    s=s.replace(',','')
    return float(s)
def sig(s):
    s=s.replace(',','').lower().split('e')[0].lstrip('0').replace('.','').lstrip('0')
    return max(1,len(s))
out=[];tot=0;bycat=collections.Counter();badcat=collections.Counter()
for r in rows:
    t=r['raw']
    for m in pat.finditer(t):
        a,op,b,eq,c=m.groups()
        # skip if preceded by an operator (part of a longer expression)
        pre=t[max(0,m.start()-4):m.start()]
        if re.search(r'[\+\-\*/×·^(]\s*$',pre) or 'times' in pre or 'cdot' in pre: continue
        try: A,B,C=f(a),f(b),f(c)
        except: continue
        if op in('+',): v=A+B
        elif op=='-': v=A-B
        elif op in('/','÷'):
            if B==0: continue
            v=A/B
        else: v=A*B
        tot+=1; bycat[r['category']]+=1
        sg=sig(c)
        if v==0: ok=abs(C)<1e-9
        else:
            rel=abs(C-v)/abs(v)
            tol=0.6*10**(-(sg-1)) if eq!='=' or '.' in c or True else 0
            # exact for integer results with '='
            if eq=='=' and float(v).is_integer() and '.' not in c and 'e' not in c.lower(): ok=(C==v)
            else: ok= rel<=max(tol,1e-9)*1.0 or abs(round(v,len(c.split('.')[1]) if '.' in c else 0)-C)<1e-9
        if not ok:
            badcat[r['category']]+=1
            out.append(dict(id=r['id'],sample=r['sample'],cat=r['category'],correct=r['correct'],cut=r['cut'],pos=m.start(),frac=m.start()/len(t),expr=m.group(0),true=v,ctx=t[max(0,m.start()-80):m.end()+80]))
print('checked',tot,'bad',len(out)); print(bycat); print(badcat)
json.dump(out,open('work/arith_bad.json','w'))
for o in out[:int(sys.argv[1]) if len(sys.argv)>1 else 60]:
    print(o['id'],o['sample'],o['cat'][:5],'ok' if o['correct'] else ('cut' if o['cut'] else 'WRONG'),'%.2f'%o['frac'],'|',o['expr'].encode('ascii','replace').decode(),'| true %.6g'%o['true'])
