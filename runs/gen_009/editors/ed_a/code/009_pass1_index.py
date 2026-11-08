# Hypothesis: token-level labels (eval sweep-end breaks, other breaks, paragraph-initial Hence, forced-chain positions) can be indexed in pass-1 rows.
import json, numpy as np
from wsp.editor import *
D=json.load(open('work/pass1_plan.json')); S=json.load(open('work/sweeps.json'))
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')}
M=json.load(open('work/fwd_1/meta.json'))
nt=M['n_tokens']; off=np.concatenate([[0],np.cumsum(nt)])
FORCE="Hence answer is ("
rows=[]  # (global_row, text_idx, qid, kind, extra)
seen=set()
for ti,(t,m) in enumerate(zip(D['texts'],D['meta'])):
    enc=tok(t,return_offsets_mapping=True,add_special_tokens=False)
    ids=enc['input_ids']; om=enc['offset_mapping']; assert len(ids)==nt[ti],(ti,len(ids),nt[ti])
    qid=m['id']; P=render_think(qid); t0=len(P)
    ends=np.array([o[1] for o in om])
    def tok_at(c):  # token containing char c
        return int(np.searchsorted(ends,c,side='right'))
    think_tok0=tok_at(t0)
    plen=len(t)-len(FORCE) if m['kind']=='probe' else len(t)
    evs=[x for x in S[qid] if x['eval']]; lis=[x for x in S[qid] if not x['eval']]
    sw_br=set()
    for x in evs:
        c=t0+x['end']-len('<think>')+2
        if c<=plen: sw_br.add(tok_at(c-1))
    lis_br=set()
    for x in lis:
        c=t0+x['end']-len('<think>')+2
        if c<=plen: lis_br.add(tok_at(c-1))
    for i in range(think_tok0,len(ids)):
        s=t[om[i][0]:om[i][1]]
        if om[i][1]>plen: # forced region
            j=i-tok_at(plen-1)  # 1=Hence,2=answer,3=is,4=(
            rows.append((int(off[ti]+i),ti,qid,'forced%d'%j,m['cut'])); continue
        key=(qid,om[i][0])
        dup = key in seen; seen.add(key)
        kind='tok'
        if '\n\n' in s: kind='brk'
        if i in sw_br: kind='swend'
        elif i in lis_br: kind='lisend'
        prev=t[om[i-1][0]:om[i-1][1]]
        if s=='Hence' and prev.endswith('\n\n'): kind='hence_init'
        rows.append((int(off[ti]+i),ti,qid,kind+('_dup' if dup else ''),m['cut']))
    # prompt rows
    for i in range(0,think_tok0):
        rows.append((int(off[ti]+i),ti,qid,'prompt%d'%min(i,9),m['cut']))
import collections
print(collections.Counter(r[3] for r in rows).most_common(30))
json.dump(rows,open('work/p1_rows.json','w'))
