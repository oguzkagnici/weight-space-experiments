# Hypothesis: paragraph breaks right before a 2nd/3rd option sweep can be labelled from the text; build token-level labels for pass-1 texts.
# Output work/labels_p1.npz: per global token row: text idx, is_break, is_pos (break before sweep>=2), is_first (break before sweep 1), in_think, char frac.
import json, re, numpy as np, sys
from wsp.editor import *
SW=re.compile(r"^(?:Now,? |So,? |Thus,? |But |Better |We |OK,? )?(?:let'?s|let’s|let us|we(?:'ll| will| need to| should| can| must)?)\s+(?:re-?)?(?:examine|analy[sz]e|evaluate|go through|check|consider|review|look at|assess|test|parse|re-?evaluate|re-?examine|reconsider|scan|list)\s+(?:each|all|the|every|them|options|again|candidate)",re.I)
FIRST=re.compile(r"^(?:[-*]\s*)?(?:\*\*)?(?:Option\s+A\b|\(A\)|A[.:)]\s)")
B={}
for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl'):
    b=json.loads(l); B[b['id']]=b
ids=json.load(open('work/p1_ids.json'))
fwd=sys.argv[1] if len(sys.argv)>1 else None
meta=json.load(open(f'{fwd}/meta.json')) if fwd else None
rows=[]
for ti,qid in enumerate(ids):
    b=B[qid]; text=trace_text(b)
    enc=tok(text, return_offsets_mapping=True); offs=enc['offset_mapping']; tids=enc['input_ids']
    if meta: assert meta['n_tokens'][ti]==len(tids), (ti, meta['n_tokens'][ti], len(tids))
    t0=text.find('<think>')+len('<think>'); t1=text.find('</think>'); t1=len(text) if t1<0 else t1
    # sweep starts (char positions in text)
    starts=[]
    for m in re.finditer(r'\n\n', text[t0:t1]):
        s=t0+m.end(); p=text[s:s+200].strip()
        if SW.match(p) or FIRST.match(p):
            if not starts or s-starts[-1]>300: starts.append(s)
    first=starts[:1]; pos=starts[1:]
    for k,(a,e) in enumerate(offs):
        piece=text[a:e]
        isb = ('\n\n' in piece) and t0<=a<t1
        nxt = e  # char after this token
        lab_pos = isb and any(abs(nxt-s)<=2 for s in pos)
        lab_first = isb and any(abs(nxt-s)<=2 for s in first)
        near = any(0<nxt-s<=300 for s in starts)   # inside the start of a sweep (ambiguous)
        rows.append((ti,int(isb),int(lab_pos),int(lab_first),int(near),int(t0<=a<t1),(a-t0)/max(1,t1-t0),b['cut'],b['correct']))
R=np.array(rows,dtype=np.float32)
np.save('work/labels_p1.npy',R)
print('tokens',len(R),'breaks',int(R[:,1].sum()),'pos',int(R[:,2].sum()),'first',int(R[:,3].sum()),'pos in cut',int(R[(R[:,2]==1)&(R[:,7]==1)].shape[0]))
