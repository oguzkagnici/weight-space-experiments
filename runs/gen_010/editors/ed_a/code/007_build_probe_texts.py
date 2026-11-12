# Hypothesis: at a two-finalist decision the letter distribution carries an early-letter prior that decides near-ties; build forced-commit probe texts right after the last pair statement (and at natural first-commit letters) for a read pass.
import json, re, collections, numpy as np
from wsp.editor import *
pat=re.compile(r"(?:between|either)\s*\(?([A-J])\)?\s*(?:,|and|or)\s*\(?([A-J])\)?(?![A-Za-z])|\(?\b([A-J])\)?\s+(?:or|vs\.?|versus)\s+\(?([A-J])\)?(?![A-Za-z])")
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
rows+= [dict(json.loads(l),sample=-1) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
SUF="\n\nThus I'll go with ("
items=[]
for r in rows:
    t=r['think']; assert t.startswith('<think>')
    fs=[]
    for m in pat.finditer(t):
        a,b=(m.group(1),m.group(2)) if m.group(1) else (m.group(3),m.group(4))
        if a!=b: fs.append((a,b,m.end()))
    if not fs: continue
    a,b,e=fs[-1]
    nl=t.find('\n',e); nl=len(t) if nl<0 else nl
    body=t[len('<think>'):nl]
    text=render_think(r['id'])+body+SUF
    items.append(dict(id=r['id'],sample=r['sample'],gold=r['gold'],pred=r['pred'],correct=r['correct'],cut=r['cut'],a=a,b=b,rel=nl/len(t),text=text))
for it in items: it['ntok']=len(tok.encode(it['text'],add_special_tokens=False))
print(len(items), 'total tokens', sum(i['ntok'] for i in items), 'max', max(i['ntok'] for i in items))
c=collections.Counter((i['cut'], i['gold'] in (i['a'],i['b'])) for i in items); print(c)
json.dump(items, open('work/probe_items_all.json','w'))
