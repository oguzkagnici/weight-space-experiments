# Hypothesis: the pass@4 bank has enough questions with both right and finished-wrong samples (mixed) to compare stem re-statement within question.
import json, collections
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
def load_rows():
    rows=[json.loads(l) for l in open(S+'passk_gen008champ_cap8192_det1.jsonl')]
    have={(r['id'],r['sample']) for r in rows}
    for l in open(S+'trace_bank_gen008champ_cap8192_det1.jsonl'):
        r=json.loads(l); r.setdefault('sample',0)
        if (r['id'],r['sample']) not in have: rows.append(r)
    return rows
if __name__=='__main__':
    rows=load_rows(); print(len(rows), rows[0].keys())
    byq=collections.defaultdict(list)
    for r in rows: byq[r['id']].append(r)
    QUANT={'business','chemistry','physics','math','engineering'}
    n=collections.Counter()
    for q,rs in byq.items():
        pat=''.join('-' if r['cut'] else ('R' if r['correct'] else 'W') for r in sorted(rs,key=lambda r:r['sample']))
        kind='mixed' if 'R' in pat and 'W' in pat else ('allR' if set(pat)<={'R'} else ('hasR' if 'R' in pat else 'noR'))
        n[(kind,rs[0]['category'] in QUANT,len(rs))]+=1
        if kind=='mixed': print(q,rs[0]['category'],pat,'gold',rs[0]['gold'],'preds',''.join(r['pred'] or '-' for r in sorted(rs,key=lambda r:r['sample'])),[r['tokens'] for r in rs])
    for k,v in sorted(n.items(),key=str): print(k,v)
