# Hypothesis: truncated traces waste budget re-copying the option list verbatim (and re-listing it several times), more than finished traces.
import json, re, difflib
from wsp.editor import *
Q = questions()
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')]
def norm(s): return re.sub(r'\W+',' ',s.lower()).strip()
out=[]
for r in rows:
    q=Q[r['id']]; opts=[norm(o) for o in q['options']]
    lines=[l for l in r['think'].split('\n') if l.strip()]
    copy_chars=0; copy_lines=0; blocks=0; prev=False
    for l in lines:
        m=re.match(r'^\s*[-*]?\s*\(?([A-J])[\).:]\s*(.*)$', l)
        is_copy=False
        if m:
            idx=ord(m.group(1))-65
            body=norm(m.group(2))
            if idx < len(opts) and len(opts[idx])>0:
                ratio=difflib.SequenceMatcher(None, body[:len(opts[idx])+20], opts[idx]).ratio()
                if ratio>0.8: is_copy=True
        if is_copy:
            copy_chars+=len(l); copy_lines+=1
            if not prev: blocks+=1
        prev=is_copy
    out.append((r['id'],r['category'],r['cut'],r['correct'],len(r['think']),copy_lines,blocks,copy_chars))
import numpy as np
for cut in (0,1):
    s=[o for o in out if o[2]==cut]
    fr=np.array([o[7]/o[4] for o in s]); bl=np.array([o[6] for o in s])
    print('cut',cut,'n',len(s),'mean copy-char share %.3f'%fr.mean(),'traces with >=1 block',(bl>=1).sum(),'>=2 blocks',(bl>=2).sum(),'mean blocks %.2f'%bl.mean())
for o in sorted(out,key=lambda o:-o[7])[:15]: print(o)
