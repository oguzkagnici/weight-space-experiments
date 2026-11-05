# Hypothesis: after a short "conclusion" paragraph (Thus/So/Therefore/Hence ... letter), finished traces commit (final answer line) while truncated traces open a new reconsideration paragraph; count these decision points on DEV traces and the first words that follow.
import json, re, random, collections
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl')]
# DEV/HOLDOUT split of the 96 bank questions, stratified by cut status
random.seed(4)
cut=[r['id'] for r in rows if r['cut']==1]; fin=[r['id'] for r in rows if r['cut']==0]
random.shuffle(cut); random.shuffle(fin)
hold=cut[:18]+fin[:14]
dev=[r['id'] for r in rows if r['id'] not in hold]
json.dump({'dev':dev,'holdout':hold},open('work/split.json','w'))
print('dev',len(dev),'holdout',len(hold))
concl=re.compile(r'^(Thus|So|Therefore|Hence)\b[^\n]{0,160}?(\b[A-J]\b|\([A-J]\))[^\n]{0,60}$')
final=re.compile(r'answer is \([A-J]\)|[Ff]inal|output|[Ww]e (should|will|need to) (present|output|give|answer|respond|produce|write)')
D=[]
firsts=collections.Counter(); byc=collections.Counter()
for r in rows:
    if r['id'] not in dev: continue
    t=r['think']
    paras=t.split('\n\n')
    pos=0
    for i,p in enumerate(paras[:-1]):
        pos_end=pos+len(p)  # index of '\n\n'
        if concl.match(p.strip()) and len(p)<260 and not final.search(p):
            nxt=paras[i+1]
            lab='commit' if final.search(nxt[:200]) else 'reconsider'
            fw=' '.join(nxt.split()[:2])
            firsts[(lab,nxt.split()[0] if nxt.split() else '')]+=1
            byc[(r['cut'],lab)]+=1
            D.append({'id':r['id'],'cut':r['cut'],'correct':r['correct'],'char':pos_end+2,'frac':(pos_end+2)/len(t),'label':lab,'para':p[:120],'next':nxt[:80]})
        pos=pos_end+2
print(len(D), byc)
for k,v in firsts.most_common(30): print(v,k)
json.dump(D,open('work/decisions_dev.json','w'))
import numpy as np
for lab in ['commit','reconsider']:
    print(lab, 'mean frac', np.mean([d['frac'] for d in D if d['label']==lab]))
for d in D[:12]: print(d['cut'],d['label'],repr(d['para'][:80]),'->',repr(d['next'][:50]))
