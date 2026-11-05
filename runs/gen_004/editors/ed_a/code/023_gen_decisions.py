# Hypothesis: in generated traces, the edit raises the commit rate at post-conclusion paragraph breaks (the targeted behaviour) without raising the number of conclusion paragraphs per 1k tokens much; compare original vs edited traces of a generate pass.
import json, re, sys, numpy as np
K=int(sys.argv[1]); d=f'work/fwd_{K:02d}'
concl=re.compile(r'^(Thus|So|Therefore|Hence)\b[^\n]{0,160}?(\b[A-J]\b|\([A-J]\))[^\n]{0,60}$')
final=re.compile(r'answer is \([A-J]\)|[Ff]inal|output|[Ww]e (should|will|need to) (present|output|give|answer|respond|produce|write)')
for m in ['original','edited']:
    R={json.loads(l)['id']:json.loads(l) for l in open(f'{d}/gen_{K}_{m}.jsonl')}
    nc=nr=0; T=0; npts=[]; after=[]
    for q,r in R.items():
        paras=r['think'].split('\n\n'); T+=int(r['tokens']); first=None; pos=0
        for i,p in enumerate(paras[:-1]):
            pos+=len(p)+2
            if concl.match(p.strip()) and len(p)<260 and not final.search(p):
                if final.search(paras[i+1][:200]): nc+=1
                else: nr+=1
                if first is None: first=pos
        if first is not None: after.append(1-first/len(r['think']))
    print(f'{m:9s} conclusion points/1k tok {1000*(nc+nr)/T:.2f}  commit {nc} reconsider {nr} commit-rate {nc/(nc+nr):.2f}  share of trace after first conclusion {np.mean(after):.2f} (n {len(after)})')
