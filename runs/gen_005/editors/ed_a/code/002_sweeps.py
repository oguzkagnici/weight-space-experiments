# Hypothesis: truncated traces repeat full option-by-option sweeps (re-evaluating all options, incl. ones already ruled out) more than finished ones; the 2nd+ sweep is the wasteful episode.
import json, re, numpy as np
from wsp.editor import *
Q = questions()
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')]
def norm(s): return re.sub(r'[^a-z0-9 ]+',' ',s.lower()).split()
def opt_of_line(l, opts):
    s=l.strip()
    m=re.match(r'^[-*\s]*(?:\*\*)?(?:Option\s+)?\(?([A-J])(?:\)|:|\.|\s*[-:=])', s)
    if m and ord(m.group(1))-65 < len(opts): return ord(m.group(1))-65
    w=norm(re.sub(r'^[-*\s]+','',s))[:3]
    if len(w)>=1:
        for i,o in enumerate(opts):
            ow=norm(o)[:3]
            if len(ow)>=1 and w[:min(2,len(ow))]==ow[:min(2,len(ow))] and len(ow[0])>3: return i
    return None
res=[]
for r in rows:
    opts=Q[r['id']]['options']
    text=r['think']
    lines=text.split('\n')
    pos=0; info=[]
    for l in lines:
        info.append((pos,l,opt_of_line(l,opts) if l.strip() else -1)); pos+=len(l)+1
    # sweeps: runs of option lines (blank lines -1 skipped), >=4 distinct options
    sweeps=[]; cur=[]
    for p,l,o in info:
        if o==-1: continue
        if o is not None: cur.append((p,o,l))
        else:
            if len(set(x[1] for x in cur))>=4: sweeps.append(cur)
            cur=[]
    if len(set(x[1] for x in cur))>=4: sweeps.append(cur)
    tot=len(text)
    sw_chars=[ (s[-1][0]+len(s[-1][2]))-s[0][0] for s in sweeps]
    res.append(dict(id=r['id'],cat=r['category'],cut=r['cut'],correct=r['correct'],n=len(sweeps),chars=tot,
        swfrac=sum(sw_chars)/tot, rep_frac=sum(sw_chars[1:])/tot, first=(sweeps[0][0][0]/tot if sweeps else None)))
for cut in (0,1):
    s=[x for x in res if x['cut']==cut]
    n=np.array([x['n'] for x in s])
    print('cut',cut,'N',len(s),'mean sweeps %.2f'%n.mean(),'dist',np.bincount(n),'sweep char share %.3f'%np.mean([x['swfrac'] for x in s]),'repeat-sweep share %.3f'%np.mean([x['rep_frac'] for x in s]))
import collections
for cut in (0,1):
  for cat in sorted(set(x['cat'] for x in res)):
    s=[x for x in res if x['cut']==cut and x['cat']==cat]
    if s: print(cut,cat,len(s),[x['n'] for x in s])
json.dump(res,open('work/002_sweeps.json','w'))
