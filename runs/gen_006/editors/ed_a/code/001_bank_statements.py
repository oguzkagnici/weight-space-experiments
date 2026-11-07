# Hypothesis: in cut traces, mid-trace answer statements are mostly hedged and less often correct than firm ones; map them.
import json, re
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
print(len(B), sum(r['cut'] for r in B), sum(r['correct'] for r in B))
pat=re.compile(r'answer is:?\s*\(?\**\(?([A-J])\)', re.I)
pat2=re.compile(r'answer(?: is|:)?\s*\(?([A-J])\b')
for r in B:
    th=r['think']; n=len(th)
    ms=list(pat.finditer(th))
    if not ms: continue
    if r['cut']:
        s=[]
        for m in ms:
            ctx=th[max(0,m.start()-60):m.end()+80].replace('\n','/')
            s.append(f"  {m.start()/n:.2f} {m.group(1)}{'*' if m.group(1)==r['gold'] else ' '} ...{ctx}")
        print(r['id'],r['category'],'gold',r['gold'],'pred',r['pred'],'tok',r['tokens'],'nstat',len(ms))
        print('\n'.join(s[:4]))
