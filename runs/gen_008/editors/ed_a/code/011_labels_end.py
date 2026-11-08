# Hypothesis: sweep-end breaks (the paragraph break right after the paragraph holding the LAST option's evaluation) are a sharp,
# lexically defined state; label them per token for pass-1 texts (col 2 = sweep-end break, col 3 = first sweep end of the trace,
# col 4 = next paragraph already commits).
import json, re, numpy as np
from wsp.editor import *
B={}
for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl'):
    b=json.loads(l); B[b['id']]=b
Q=questions(); ids=json.load(open('work/p1_ids.json'))
COMMIT=re.compile(r"^(Thus|Hence|Therefore|So)\b.{0,120}?(answer|choose|pick|option [A-J]\b|\([A-J]\))",re.I|re.S)
rows=[]; ends_all={}
for ti,qid in enumerate(ids):
    b=B[qid]; text=trace_text(b); nopt=len(Q[qid]['options']); L='ABCDEFGHIJ'[nopt-1]
    pm=re.compile(r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:Option\s+%s\b|\(%s\)|%s[.:)]\s)"%(L,L,L))
    enc=tok(text, return_offsets_mapping=True); offs=enc['offset_mapping']
    t0=text.find('<think>')+7; t1=text.find('</think>'); t1=len(text) if t1<0 else t1
    seg=text[t0:t1]; ends=[]; pstart=0
    brks=[m.end()+t0 for m in re.finditer(r'\n\n',seg)]
    starts=[t0]+brks
    for i in range(len(brks)):
        p=text[starts[i]:brks[i]]; nxt=text[brks[i]:(brks[i+1] if i+1<len(brks) else t1)]
        if pm.search(p) and not pm.search(nxt): ends.append((brks[i], bool(COMMIT.match(nxt.strip()))))
    ends_all[qid]=[(e,c) for e,c in ends]
    E={e:c for e,c in ends}; firstE=ends[0][0] if ends else -1
    for k,(a,e) in enumerate(offs):
        isb=('\n\n' in text[a:e]) and t0<=a<t1
        hit=[x for x in E if isb and abs(e-x)<=2]
        rows.append((ti,int(isb),int(bool(hit)),int(bool(hit) and abs(e-firstE)<=2),int(bool(hit) and E[hit[0]]),int(t0<=a<t1),0,b['cut'],b['correct']))
R=np.array(rows,dtype=np.float32); np.save('work/labels_end_p1.npy',R); json.dump(ends_all,open('work/ends_p1.json','w'))
print('breaks',int(R[:,1].sum()),'sweep-end breaks',int(R[:,2].sum()),'first',int(R[:,3].sum()),'followed by commit',int(R[:,4].sum()))
