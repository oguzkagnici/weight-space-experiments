# Hypothesis: (bookkeeping) repartition the pass-1/2 texts into three read passes under the 200k-token limit: p1 = NON/CUE (late+early, wrong samples), p2 = INF/BASE, p3 = judging positions + early arms on cut/right samples + overflow.
import json, sys, importlib, collections
sys.path.insert(0,'code'); b=importlib.import_module('005_build_ins')
import runpy
ns=runpy.run_path('code/006_build_p12.py')      # rebuilds P1, P2 lists (prints its summary)
P1,P2,add,rows,cuts,ins,fit,ntok=ns['P1'],ns['P2'],ns['add'],ns['rows'],b.cuts,ns['ins'],ns['fit'],b.ntok
# late BASE texts were dropped from the saved p2 but are still in P2
A=[x for x in P1]; B=[x for x in P2 if x[1]['kind']=='ins' and x[1]['status']=='wrong']; C=[x for x in P2 if not (x[1]['kind']=='ins' and x[1]['status']=='wrong')]
LIM=194000
def trim(P,C):
    P=sorted(P,key=lambda x:(x[1]['cut']!='late',x[1]['id'],x[1]['sample']))   # keep late first; move early items of the last questions to p3
    while sum(ntok(t) for t,_ in P)>LIM:
        # move a whole (id, sample, cut=early) group
        last=P[-1][1]; grp=[x for x in P if x[1]['cut']=='early' and x[1]['id']==last['id'] and x[1]['sample']==last['sample']]
        if not grp: grp=[P[-1]]
        for g in grp: P.remove(g); C.append(g)
    return P
A=trim(A,C); B=trim(B,C)
for name,P in (('p1',A),('p2',B),('p3',C)):
    tot=sum(ntok(t) for t,_ in P)
    print(name,'texts',len(P),'tokens',tot,collections.Counter((mt['kind'],mt.get('cut'),mt.get('arm'),mt.get('status')) for _,mt in P))
    json.dump([mt for _,mt in P],open('work/%s_meta.json'%name,'w')); json.dump([t for t,_ in P],open('work/%s_texts.json'%name,'w'))
    json.dump({"type":"read","model":"original","texts":[t for t,_ in P],"return":{"token_logprobs":True,"topk":50,"positions":"last","attn_last":[2,4,6,8,10,12,14,16,18,20,22]}},open('work/req_%s.json'%name,'w'))
