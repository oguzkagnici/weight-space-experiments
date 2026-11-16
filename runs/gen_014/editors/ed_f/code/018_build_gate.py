# Hypothesis (read gate): the edited model puts more attention on the stem at HELD-OUT option-judging positions (edited L22 heads), little more at background positions; measure what that does to the next-token distribution and to forced conclusions.
import json, sys, numpy as np, collections
from wsp.editor import *
ntok=lambda s:len(tok.encode(s,add_special_tokens=False))
m3=json.load(open('work/p3b_meta.json')); t3=json.load(open('work/p3b_texts.json'))
P=[(t,dict(m,src='p3',idx=i)) for i,(t,m) in enumerate(zip(t3,m3)) if m['kind'] in ('judge','judge2','bg')]
m2=json.load(open('work/p2_meta.json')); t2=json.load(open('work/p2_texts.json'))
resp=['mmlupro_2654','mmlupro_6110','mmlupro_6614','mmlupro_6800']
base=[(t,dict(m,src='p2',idx=i)) for i,(t,m) in enumerate(zip(t2,m2)) if m['arm']=='BASE' and m['cut']=='late']
base.sort(key=lambda x:(x[1]['id'] not in resp,ntok(x[0])))
tot=sum(ntok(t) for t,_ in P)
for x in base:
    if tot+ntok(x[0])>194000: continue
    P.append(x); tot+=ntok(x[0])
print('texts',len(P),'tokens',tot,collections.Counter((mt['kind'],mt.get('arm')) for _,mt in P),'BASE questions',collections.Counter(mt['id'][8:] for _,mt in P if mt['kind']=='ins'))
json.dump([mt for _,mt in P],open('work/p5_meta.json','w')); json.dump([t for t,_ in P],open('work/p5_texts.json','w'))
json.dump({"type":"read","model":"edited","texts":[t for t,_ in P],"return":{"token_logprobs":True,"topk":50,"positions":"last","attn_last":[22]}},open('work/req_p5.json','w'))
