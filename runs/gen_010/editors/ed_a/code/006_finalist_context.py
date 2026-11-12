# Hypothesis: in wrong picks with gold as the later finalist, the decision is made at a short commit phrase (letter token) right after the pair is named; show context.
import json, re, collections
exec(open('code/005_finalists.py').read().split("st=collections.Counter()")[0])
for r in rows:
    if r['cut'] or not r['pred'] or r['correct']: continue
    fs=[]
    for m in pat.finditer(r['think']):
        a,b=(m.group(1),m.group(2)) if m.group(1) else (m.group(3),m.group(4))
        if a!=b: fs.append((a,b,m.start()))
    if not fs: continue
    a,b,p=fs[-1]
    if r['gold'] in (a,b) and r['pred'] in (a,b):
        t=r['think']
        print('====',r['id'],r.get('sample'),'gold',r['gold'],'pred',r['pred'],'pos',round(p/len(t),2))
        print(t[max(0,p-250):p+550].replace('\n',' | ').encode('ascii','replace').decode())
