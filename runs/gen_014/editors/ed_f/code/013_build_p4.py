# Hypothesis (pass 4): on cue questions the model fails because it does not hold the implication of the cue (false belief), not because it fails to use a belief it has: in a two-statement forced choice (right implication vs the premise of its wrong pick) it prefers the wrong statement on never-right questions, bare and inside its own trace. Also: a cue quote does not damage right conclusions.
import json, sys, re, importlib, collections, numpy as np
sys.path.insert(0,'code'); m=importlib.import_module('001_overview'); b=importlib.import_module('005_build_ins'); p6=None
from wsp.editor import *
Q=questions(); rows=m.load_rows(); ntok=b.ntok
KN={
'mmlupro_1196':('A temporary restraining order that is extended beyond the 14-day limit is treated as a preliminary injunction and can be appealed.','An order that extends a temporary restraining order cannot be appealed, because it is not a final judgment.'),
'mmlupro_1275':('An ordinance that makes picketing depend on the permission of the mayor, with no standards for granting it, is unconstitutional on its face.','A union that pickets without the permission that a city ordinance requires can be enjoined for violating the ordinance.'),
'mmlupro_1409':('When an ordinance that is already in force blocks a planned sale, a mootness defense is weaker than a ripeness defense.','When an ordinance that is already in force blocks a planned sale, a ripeness defense is weaker than a mootness defense.'),
'mmlupro_1445':('A total ban on advertising a lawful product is more extensive than necessary and violates the First Amendment.','A total ban on advertising a lawful product is constitutional when it serves a legitimate interest in public health.'),
'mmlupro_1643':('A videotape made two months after the judgment is not newly discovered evidence, because such evidence must have existed at the time of the trial.','A videotape made two months after the judgment is newly discovered evidence that supports a motion for relief from the judgment.'),
'mmlupro_1705':('Seeing the occupants of a car fidget nervously does not give the police probable cause to search the car.','Seeing the occupants of a car fidget nervously gives the police probable cause to search the car.'),
'mmlupro_2654':("Waking up shaking and sweating from a dream of falling to one's death is a nightmare.","Waking up shaking and sweating from a dream of falling to one's death is sleep paralysis."),
'mmlupro_4811':('The European colonization of the interior of Africa took place after 1750.','The European colonization of the interior of Africa took place between 1450 and 1750.'),
'mmlupro_6110':('A study that assigns patients to successive dose cohorts, with every patient knowing the drug, is an open-label clinical trial.','A study that assigns patients to successive dose cohorts, with every patient knowing the drug, is a randomized clinical trial.'),
'mmlupro_6562':('Timolol is a non-selective beta blocker, and betaxolol is beta-1 selective.','Betaxolol is a non-selective beta blocker, and timolol is beta-1 selective.'),
'mmlupro_6614':('For recurrent low back pain in an obese patient with a normal examination, the next step is a weight loss program rather than imaging or physical therapy.','For recurrent low back pain in an obese patient with a normal examination, the next step is imaging or physical therapy rather than a weight loss program.'),
'mmlupro_6800':('The most common bacterial cause of acute maxillary sinusitis is Streptococcus pneumoniae.','The most common bacterial cause of acute maxillary sinusitis is Mycoplasma pneumoniae.'),
'mmlupro_6829':('If the world price of a good is above the domestic price, the country will be a net exporter of the good.','If the world price of a good is above the domestic price, the country will import more of the good.'),
}
P=[]
HEAD='<|startoftext|><|im_start|>user\n'; TAIL='\n\nAnswer with the letter of the correct option.<|im_end|>\n<|im_start|>assistant\n<think></think>The answer is ('
assert render('mmlupro_2654').startswith(HEAD) and render('mmlupro_2654').endswith(TAIL)
for q,(r,w) in KN.items():
    for order in (0,1):
        s1,s2=(r,w) if order==0 else (w,r)
        P.append((HEAD+'Which of the following statements is correct?\n\n(A) %s\n(B) %s'%(s1,s2)+TAIL,dict(kind='know',ctx='bare',id=q,order=order)))
byq=collections.defaultdict(list)
for r in rows: byq[r['id']].append(r)
for r in rows:
    if r['id'] not in KN: continue
    T,off,think,early,late=b.cuts(r); st='cut' if r['cut'] else ('right' if r['correct'] else 'wrong'); rr,w=KN[r['id']]
    if early:
        for order in (0,1):
            s1,s2=(rr,w) if order==0 else (w,rr)
            P.append((T[:off+early]+'Before going through the options, which of these two statements is correct?\n(1) %s\n(2) %s\nThe correct statement is ('%(s1,s2),dict(kind='know',ctx='trace',id=r['id'],sample=r['sample'],status=st,order=order)))
    if st=='right' and late:
        wp=[x['pred'] for x in byq[r['id']] if not x['cut'] and not x['correct'] and x['pred']]; modal=collections.Counter(wp).most_common(1)[0][0]
        cut=late
        if ntok(T[:off+cut])>3850:
            br=[mm.end() for mm in re.finditer(r'\n\n',think) if mm.end()<cut]; cut=[c for c in br if ntok(T[:off+c])<=3850][-1]
        c,n,i=b.ANN[r['id']]
        for arm,ins in (('BASE',''),('NON',(b.TPL%n if n else b.TPLF%b.FILL)+'\n\n'),('CUE',b.TPL%c+'\n\n')):
            P.append((T[:off+cut]+ins+b.PROBE,dict(kind='ins',cut='late',arm=arm,id=r['id'],sample=r['sample'],status='right',gold=r['gold'],pred=r['pred'],wrong=modal)))
tot=sum(ntok(t) for t,_ in P); print('texts',len(P),'tokens',tot,collections.Counter((mt['kind'],mt.get('ctx'),mt.get('arm'),mt.get('status')) for _,mt in P))
assert tot<197000
json.dump([mt for _,mt in P],open('work/p4_meta.json','w')); json.dump([t for t,_ in P],open('work/p4_texts.json','w'))
json.dump({"type":"read","model":"original","texts":[t for t,_ in P],"return":{"token_logprobs":True,"topk":50,"positions":"last"}},open('work/req_p4.json','w'))
print(tok.encode('(1',add_special_tokens=False),tok.encode('(2',add_special_tokens=False),tok.encode('1',add_special_tokens=False),tok.encode('2',add_special_tokens=False))
asc=lambda t:t.encode('ascii','replace').decode(); print(repr(asc(P[0][0]))); print(repr(asc(P[30][0][-420:])))
