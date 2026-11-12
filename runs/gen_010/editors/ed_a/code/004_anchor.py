# Hypothesis: wrong answers come from an early guess (stated before the options are evaluated) that the model then keeps (anchoring); measure first-lean vs final.
import json, re, collections
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
pat=re.compile(r"(?:answer(?: is| would be| should be| =|:)?|go with|choose|pick|lean(?:s|ing)? (?:towards?|to)|likely|probably|best (?:answer|option|choice)(?: is)?|option)\s*\(?\s*([A-J])\)?(?![A-Za-z])")
st=collections.Counter()
for r in rows:
    if r['cut'] or not r['pred']: continue
    t=r['think']
    ms=[(m.start(),m.group(1)) for m in pat.finditer(t)]
    # preference statements only: exclude 'option X' listing by requiring words other than option
    pref=[(p,l) for (p,l),m in zip(ms,pat.finditer(t)) if not m.group(0).lower().startswith('option')]
    if not pref: st['nopref']+=1; continue
    p0,l0=pref[0]; rel=p0/len(t)
    key=('right' if r['correct'] else 'wrong')
    st[key]+=1
    st[key+'_first=final']+= (l0==r['pred'])
    st[key+'_first=gold']+= (l0==r['gold'])
    st[key+'_firstrel<0.3']+= rel<0.3
    st[key+'_nletters']+= len(set(l for _,l in pref))>1
print(dict(st))
