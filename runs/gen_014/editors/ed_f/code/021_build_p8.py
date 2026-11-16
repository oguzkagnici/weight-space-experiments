# Hypothesis (pass 8, replication): the per-question split of the cue-quote effect (helps on the health/psychology vignettes, firms up the wrong pick on law/history) does not depend on the wording of the insertion or on the choice of the non-cue control phrase.
import json, sys, re, importlib, collections
sys.path.insert(0,'code'); m=importlib.import_module('001_overview'); b=importlib.import_module('005_build_ins')
from wsp.editor import *
Q=questions(); rows=m.load_rows(); ntok=b.ntok
NON2={'mmlupro_1196':'The homeowner has moved to dismiss the appeal','mmlupro_1275':'to require higher certification for instructors who wish to teach in the gifted and talented classes',
'mmlupro_1409':'An out of state surfboard manufacturer had planned to sell a new line of fiberglass surfboards','mmlupro_1445':'On the basis of scientific studies showing a causal relationship',
'mmlupro_1643':'for injuries arising out of an accident involving the parties','mmlupro_1705':'women dressed as superheroes had robbed a man in his house',
'mmlupro_2654':"Quinn's experience is often categorized as a",'mmlupro_4811':'by draining African wealth and by making it impossible to develop more rapidly the resources of the continent',
'mmlupro_6110':'Six to eight patients are assigned to each dose','mmlupro_6562':'begins using eyedrops for glaucoma','mmlupro_6614':'She walks 3 miles daily for exercise',
'mmlupro_6800':'She describes the pain as 7 out of 10 in intensity','mmlupro_6829':None}
TPL2='Hold on. The question specifically says: "%s".'; FILL2='Hold on. I should think about which of these options really fits best here.'
m1=json.load(open('work/p1_meta.json')); t1=json.load(open('work/p1_texts.json')); P=[]
for t,mt in zip(t1,m1):
    if not (mt['cut']=='late' and mt['arm']=='CUE'): continue
    q=mt['id']; cue=b.ANN[q][0]; old=b.TPL%cue+'\n\n'+b.PROBE; assert t.endswith(old); pre=t[:-len(old)]
    non=NON2[q]; assert non is None or non in Q[q]['question'],q
    for arm,ins in (('CUE2',TPL2%cue),('NON2',TPL2%non if non else FILL2)):
        P.append((pre+ins+'\n\n'+b.PROBE,dict(mt,arm=arm)))
tot=sum(ntok(t) for t,_ in P); print('texts',len(P),'tokens',tot); assert tot<197000
json.dump([mt for _,mt in P],open('work/p8_meta.json','w')); json.dump([t for t,_ in P],open('work/p8_texts.json','w'))
json.dump({"type":"read","model":"original","texts":[t for t,_ in P],"return":{"token_logprobs":True,"topk":50,"positions":"last"}},open('work/req_p8.json','w'))
print(repr(P[0][0][-200:].encode('ascii','replace').decode())); print(repr(P[1][0][-200:].encode('ascii','replace').decode()))
