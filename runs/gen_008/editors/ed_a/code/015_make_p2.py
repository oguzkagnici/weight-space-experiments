# Pass-2 request (read, EDITED v1 = code/014 8 1 1 1.5): forced-commit probes at sweep ends of bank traces:
# prefix through the sweep-end break + "Hence, the answer is (". Measures (i) edited P(Hence) at the break (dose vs gate z),
# (ii) what follows "Hence", (iii) the forced letter = value of committing there, (iv) off-target logprob changes vs pass 1.
import json, re, numpy as np
from wsp.editor import *
Q=questions()
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
probes=[]
for b in B:
    text=trace_text(b); nopt=len(Q[b['id']]['options']); L='ABCDEFGHIJ'[nopt-1]
    pm=re.compile(r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:Option\s+%s\b|\(%s\)|%s[.:)]\s)"%(L,L,L))
    t0=text.find('<think>')+7; t1=text.find('</think>'); t1=len(text) if t1<0 else t1
    brks=[m.end()+t0 for m in re.finditer(r'\n\n',text[t0:t1])]; starts=[t0]+brks; k=0
    for i in range(len(brks)):
        p=text[starts[i]:brks[i]]; nxt=text[brks[i]:(brks[i+1] if i+1<len(brks) else t1)]
        if pm.search(p) and not pm.search(nxt):
            k+=1; probes.append(dict(id=b['id'],k=k,cut=b['cut'],correct=b['correct'],gold=b['gold'],pred=b['pred'],char=brks[i],
                                     frac=(brks[i]-t0)/(t1-t0),next=nxt[:60]))
print('probes',len(probes),collections.Counter() if False else '')
def nt(t): return len(tok(t)['input_ids'])
byid={b['id']:b for b in B}
sel=[];tot=0
for kk in (1,2,3,4):
    for p in probes:
        if p['k']!=kk: continue
        t=trace_text(byid[p['id']])[:p['char']]+"Hence, the answer is ("
        n=nt(t)
        if tot+n>197000: continue
        p['text']=t; p['ntok']=n; sel.append(p); tot+=n
print('selected',len(sel),'tokens',tot,'by k',[sum(1 for p in sel if p['k']==k) for k in (1,2,3,4)],'cut',sum(p['cut'] for p in sel))
json.dump([{k:v for k,v in p.items() if k!='text'} for p in sel],open('work/p2_probes.json','w'))
json.dump({"type":"read","model":"edited","texts":[p['text'] for p in sel],"return":{"token_logprobs":True,"topk":20,"positions":"all"}},open('work/req_p2.json','w'))
