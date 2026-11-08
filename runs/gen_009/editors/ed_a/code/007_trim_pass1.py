# Hypothesis: dropping the longest full cut traces keeps the pass under 195k tokens without losing probes.
import json
D=json.load(open('work/pass1_plan.json'))
T,M,N=D['texts'],D['meta'],D['ntok']
order=sorted([i for i,m in enumerate(M) if m['kind']=='full' and m['cut']],key=lambda i:-N[i])
drop=set(); tot=sum(N)
for i in order:
    if tot<=195000: break
    drop.add(i); tot-=N[i]
keep=[i for i in range(len(T)) if i not in drop]
T=[T[i] for i in keep];M=[M[i] for i in keep];N=[N[i] for i in keep]
print(len(T),sum(N),'dropped',[D['meta'][i]['id'] for i in drop])
json.dump(dict(texts=T,meta=M,ntok=N),open('work/pass1_plan.json','w'))
req={"type":"read","model":"original","texts":T,"return":{"token_logprobs":True,"topk":20,"inputs":["model.layers.23.feed_forward.w1"],"positions":"all"}}
json.dump(req,open('work/req1.json','w'))
