# Hypothesis: evaluation sweeps (option lines with a judgement appended, covering most options in order) can be detected in text; count them and the opener after each sweep end.
import json, re, collections
from wsp.editor import *
Q=questions()
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
pat=re.compile(r'^[ \t]*(?:[-*]\s*)?(?:\*\*)?(?:Option\s*)?\(?([A-J])\)(?:\*\*)?[\s:.]|^[ \t]*(?:[-*]\s*)?(?:\*\*)?Option\s*\(?([A-J])\)?\b',re.M)
LET='ABCDEFGHIJ'
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
def sweeps(r):
    t=r['think']; q=Q[r['id']]; opts=q['options']; nl=len(opts); last=LET[nl-1]
    ms=list(pat.finditer(t)); out=[]
    i=0
    # group consecutive option lines into runs that start at A and go in increasing order
    run=[]
    for m in ms:
        L=m.group(1) or m.group(2); k=LET.index(L)
        line_end=t.find('\n',m.end()); line_end=len(t) if line_end<0 else line_end
        body=t[m.end():line_end]
        if run and k==run[-1][0]+1 or (not run and k==0) or (run and k==0):
            if k==0: run=[]
            if k<nl:
                optn=norm(opts[k]); bn=norm(body)
                ev = len(bn) > len(optn)+12 or any(w in body.lower() for w in [' not ','incorrect','correct','wrong','likely','plausib','?','no.','yes','unlikely','too ','matches','fits','close'])
                run.append((k,m.start(),line_end,ev))
            if run and run[-1][0]==nl-1 and len(run)>=max(3,int(0.7*nl)):
                # sweep end = end of paragraph containing last option
                e=t.find('\n\n',run[-1][1]); 
                if e>=0:
                    nev=sum(x[3] for x in run)
                    out.append(dict(start=run[0][1],end=e,neval=nev,n=len(run),eval=nev>=0.6*len(run),opener=t[e+2:e+40]))
                run=[]
        else:
            run=[]
    return out
S={}
for r in B:
    S[r['id']]=sweeps(r)
tot=collections.Counter()
for r in B:
    s=S[r['id']]; ev=[x for x in s if x['eval']]
    tot[(r['cut'],'traces')]+=1; tot[(r['cut'],'sweeps')]+=len(s); tot[(r['cut'],'evsweeps')]+=len(ev); tot[(r['cut'],'has_ev')]+=len(ev)>0; tot[(r['cut'],'has_any')]+=len(s)>0
print(sorted(tot.items()))
op=collections.Counter()
for r in B:
    for x in S[r['id']]:
        if x['eval']: op[(r['cut'],x['opener'].split()[0] if x['opener'].split() else '')]+=1
print(op.most_common(40))
json.dump({k:v for k,v in S.items()},open('work/sweeps.json','w'))
