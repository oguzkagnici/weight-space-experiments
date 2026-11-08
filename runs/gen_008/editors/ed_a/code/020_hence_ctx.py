# Hypothesis check: in edited generations, "Hence" paragraphs inserted at sweep ends commit to an answer (what follows "Hence", how often an answer statement follows within the paragraph).
import json, re, sys, collections
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
E=[json.loads(l) for f in sys.argv[1:] for l in open(f)]
c=collections.Counter(); ans=0; n=0; ntr=0
for e in E:
    hs=[m.start() for m in re.finditer(r'\n\nHence',e['think'])]
    if hs: ntr+=1
    for h in hs:
        par=e['think'][h+2:].split('\n\n')[0]; n+=1
        c[' '.join(par.split()[:3])]+=1
        ans+=bool(re.search(r'answer|\([A-J]\)',par[:250]))
print('traces with Hence paragraphs',ntr,'of',len(E),'; Hence paragraphs',n,'with answer/letter',ans)
print(c.most_common(15))
