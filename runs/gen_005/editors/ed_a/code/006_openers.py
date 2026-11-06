# Hypothesis: cut traces differ from finished ones by paragraph openers that restart work (Alternatively / Another approach / Better approach / Maybe / Let's examine each option), per token, not by computation.
import json, re, numpy as np, collections
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')]
split=json.load(open('work/split.json')); dev=set(split['dev'])
cnt={0:collections.Counter(),1:collections.Counter()}; toks={0:0,1:0}; npar={0:0,1:0}
for r in rows:
    if r['id'] not in dev: continue
    c=r['cut']; toks[c]+=r['tokens']
    for p in re.split(r'\n+', r['think']):
        p=p.strip().lstrip('-* ').replace('’',"'")
        if not p: continue
        npar[c]+=1
        w=re.findall(r"[A-Za-z']+",p)[:2]
        k=' '.join(w[:2]) if w and w[0] in ("Let's","Let","But","We","Maybe","Thus","So","Option","Now","Actually","Wait","Another","Better","Alternatively","Hence","Therefore","However","Perhaps","I","In","The","Given","Check","Hmm","Also","If","First","Then","Compute") else (w[0] if w else '')
        cnt[c][k]+=1
print('paragraphs per 1k tok: fin %.1f cut %.1f'%(1000*npar[0]/toks[0],1000*npar[1]/toks[1]))
keys=set(k for c in cnt for k,v in cnt[c].items() if v>=6)
tab=sorted(keys,key=lambda k:-(cnt[1][k]/toks[1]-cnt[0][k]/toks[0]))
for k in tab: print('%-22s fin %5.2f  cut %5.2f  (n %d/%d)'%(k,1000*cnt[0][k]/toks[0],1000*cnt[1][k]/toks[1],cnt[0][k],cnt[1][k]))
