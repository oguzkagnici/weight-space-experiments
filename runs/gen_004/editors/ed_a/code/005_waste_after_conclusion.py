# Hypothesis: a large share of tokens in DEV traces is spent after the first conclusion paragraph; quantify per trace (finished vs cut), and characterise the cut traces with no conclusion.
import json, re, numpy as np
from wsp.editor import tok
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl')}
D=json.load(open('work/decisions_dev.json')); split=json.load(open('work/split.json'))
first={}
for d in D: first.setdefault(d['id'],d['char'])
for c in [0,1]:
    w=[];tot=[]
    for q in split['dev']:
        r=rows[q]
        if r['cut']!=c or q not in first: continue
        n_all=len(tok.encode(r['think'],add_special_tokens=False)); n_pre=len(tok.encode(r['think'][:first[q]],add_special_tokens=False))
        w.append(n_all-n_pre); tot.append(n_all)
    print('cut',c,'n',len(w),'tokens after first conclusion: mean',np.mean(w),'of',np.mean(tot))
# hedge density in no-conclusion cut traces vs others
hed=re.compile(r'\b(maybe|perhaps|Alternatively|Hmm|might|could be|possibly)\b')
for name,qs in [('cut_noconcl',[q for q in split['dev'] if rows[q]['cut']==1 and q not in first]),('cut_concl',[q for q in split['dev'] if rows[q]['cut']==1 and q in first]),('fin',[q for q in split['dev'] if rows[q]['cut']==0])]:
    h=sum(len(hed.findall(rows[q]['think'])) for q in qs); T=sum(int(rows[q]['tokens']) for q in qs)
    opt=sum(len(re.findall(r'(?:^|\n)(?:Option|\()\s*\(?[A-J]\)?',rows[q]['think'])) for q in qs)
    print(name,len(qs),'hedges/1k %.2f'%(1000*h/T),'option-lines/1k %.2f'%(1000*opt/T), 'cats',sorted(set(rows[q]['category'][:5] for q in qs)))
