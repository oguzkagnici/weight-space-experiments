# Hypothesis: many wrong answers are a wrong pick between two finalists that include gold; check how often and whether the pick follows a positional rule (first-named / earlier letter / last-discussed).
import json, re, collections
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
rows+= [dict(json.loads(l),sample=-1) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
pat=re.compile(r"(?:between|either)\s*\(?([A-J])\)?\s*(?:,|and|or)\s*\(?([A-J])\)?(?![A-Za-z])|\(?\b([A-J])\)?\s+(?:or|vs\.?|versus)\s+\(?([A-J])\)?(?![A-Za-z])")
st=collections.Counter()
for r in rows:
    if r['cut'] or not r['pred']: continue
    t=r['think']
    fs=[]
    for m in pat.finditer(t):
        a,b=(m.group(1),m.group(2)) if m.group(1) else (m.group(3),m.group(4))
        if a!=b: fs.append((a,b))
    k='right' if r['correct'] else 'wrong'
    st[k]+=1
    if not fs: continue
    a,b=fs[-1]
    st[k+'_hasfin']+=1
    if r['pred'] in (a,b):
        st[k+'_pred_in_last_pair']+=1
        st[k+'_pick_first_named']+= r['pred']==a
        st[k+'_pick_earlier_letter']+= r['pred']==min(a,b)
    st[k+'_gold_in_last_pair']+= r['gold'] in (a,b)
print(dict(st))
print('--- gold in last pair, pred in pair: by pick position')
c=collections.Counter(); qs=collections.defaultdict(list)
for r in rows:
    if r['cut'] or not r['pred']: continue
    fs=[]
    for m in pat.finditer(r['think']):
        a,b=(m.group(1),m.group(2)) if m.group(1) else (m.group(3),m.group(4))
        if a!=b: fs.append((a,b))
    if not fs: continue
    a,b=fs[-1]
    if r['gold'] in (a,b) and r['pred'] in (a,b):
        gold_earlier = r['gold']==min(a,b)
        c[('gold_earlier' if gold_earlier else 'gold_later', 'right' if r['correct'] else 'wrong')]+=1
        qs[r['id']].append((gold_earlier, r['correct']))
print(c); print('questions', len(qs))
for q,v in sorted(qs.items()): print(q, v)
