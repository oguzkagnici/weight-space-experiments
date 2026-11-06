# Hypothesis: at paragraph breaks, the model's next-token distribution and the MLP-input state separate "churn" restarts (Alternatively/Maybe/Perhaps...) from other continuations, and churn-predicting breaks are concentrated in traces that get cut.
import json, re, numpy as np
from wsp.editor import *
K=1
D='work/fwd_01'
meta=json.load(open(f'{D}/meta.json'))
ids_list=json.load(open('work/req1_ids.json'))
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
V=128000
dec=lambda i: tok.decode([int(i)])
# paragraph-break tokens
brk=set()
for i in range(V):
    s=tok.decode([i])
    if s.endswith('\n\n') or s.endswith('\n\n ') : brk.add(i)
print('break tokens', len(brk))
offs=np.load(f'{D}/offsets.npy')
topi=np.load(f'{D}/topk_ids.npy'); topv=np.load(f'{D}/topk_logprobs.npy')
tlp=np.load(f'{D}/token_logprobs.npy')
CHURN=re.compile(r"^(Alternatively|Maybe|Perhaps|But perhaps|But maybe|Or maybe|Or perhaps|Another|Better approach|Better to|Possibly|Could it be|Wait, maybe|Actually maybe|Actually, maybe)")
COMMIT=re.compile(r"^(Thus|Hence|Therefore|So)\b[^\n]{0,40}(answer|final|choose|pick|go with|I'll|correct option|is \(?[A-J]\b|\([A-J]\))|^(I'll|I will|I'd) (go|choose|pick|answer)|^Final|^Answer")
recs=[]
tokpos=0
for ti,qid in enumerate(ids_list):
    r=rows[qid]; text=trace_text(r)
    ids=tok(text,add_special_tokens=False).input_ids
    assert len(ids)==meta['n_tokens'][ti]
    # prompt length: tokens of render_think
    pl=len(tok(render_think(qid),add_special_tokens=False).input_ids)
    o=int(offs[ti])
    for t in range(pl, len(ids)-1):
        if ids[t] in brk:
            nxt=tok.decode(ids[t+1:t+12]).lstrip()
            cls='churn' if CHURN.match(nxt) else ('commit' if COMMIT.match(nxt) else 'other')
            recs.append(dict(ti=ti,qid=qid,cut=r['cut'],correct=r['correct'],t=t,row=o+t,frac=(t-pl)/max(1,len(ids)-pl),cls=cls,nxt=ids[t+1],nxt_text=nxt[:30]))
print('breaks',len(recs))
import collections
c=collections.Counter((x['cut'],x['cls']) for x in recs); print(sorted(c.items()))
json.dump(recs,open('work/009_breaks.json','w'))
# next-token probability of churn first tokens at breaks
first_tok=collections.Counter(x['nxt'] for x in recs if x['cls']=='churn')
print('churn first tokens', [(dec(i),n) for i,n in first_tok.most_common(12)])
ftc=collections.Counter(x['nxt'] for x in recs if x['cls']=='commit')
print('commit first tokens', [(dec(i),n) for i,n in ftc.most_common(12)])
