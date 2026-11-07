# Hypothesis: (generate analysis) the edit leaves most thinking traces byte-identical; gains come from closed-but-cut traces that now finish with a short answer.
import json, sys, numpy as np
k=int(sys.argv[1]); D=f'work/fwd_{k:02d}'
O={json.loads(l)['id']:json.loads(l) for l in open(f'{D}/gen_{k:02d}_original.jsonl')} if False else None
import glob
fo=glob.glob(f'{D}/gen_*_original.jsonl')[0]; fe=glob.glob(f'{D}/gen_*_edited.jsonl')[0]
O={json.loads(l)['id']:json.loads(l) for l in open(fo)}; E={json.loads(l)['id']:json.loads(l) for l in open(fe)}
ident=0; fd=[]; rows=[]
for q in O:
    o,e=O[q],E[q]
    a,b=o['raw'],e['raw']
    ka=a.find('</think>'); kb=b.find('</think>')
    ta=a[:ka] if ka>=0 else a; tb=b[:kb] if kb>=0 else b
    same_think = ta==tb and (ka>=0)==(kb>=0)
    i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    if same_think: ident+=1
    else: fd.append(i/max(1,len(ta)))
    flag=('+' if (e['correct'] and not o['correct']) else '-' if (o['correct'] and not e['correct']) else ' ')
    rows.append((flag,q,o['category'][:6],'think-same' if same_think else f'diff@{i}({i/max(1,len(ta)):.2f})',f"o:{o['pred']}{'C' if o['cut'] else ''}{o['tokens']}",f"e:{e['pred']}{'C' if e['cut'] else ''}{e['tokens']}",o['gold'],repr(e['content'][:60].encode('ascii','replace').decode())))
for r in sorted(rows): print(*r)
n=len(O); g=sum(r[0]=='+' for r in rows); l=sum(r[0]=='-' for r in rows)
print(f'n {n} acc o {sum(O[q]["correct"] for q in O)} e {sum(E[q]["correct"] for q in E)} | +{g}/-{l} | think identical {ident}/{n} | first-diff frac median {np.median(fd) if fd else None}')
print('cut o',sum(O[q]['cut'] for q in O),'e',sum(E[q]['cut'] for q in E),'| tokens o',np.mean([O[q]['tokens'] for q in O]),'e',np.mean([E[q]['tokens'] for q in E]))
