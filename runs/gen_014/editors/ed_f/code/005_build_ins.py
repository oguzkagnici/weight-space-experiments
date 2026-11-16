# Hypothesis (step 2, causal): in wrong samples, quoting the decisive stem cue before the conclusion moves the forced conclusion toward gold more than quoting an equal-length non-decisive stem phrase.
import json, sys, re, importlib, collections, numpy as np
sys.path.insert(0,'code'); m=importlib.import_module('001_overview')
from wsp.editor import *
Q=questions(); rows=m.load_rows()
# hand annotation (made from the stems + gold + the wrong picks only, before any pass): cue = decisive stem phrase, non = stem phrase that does not decide, inf = the implication of the cue (oracle arm, names no option)
ANN={
'mmlupro_1196':('the court extended the TRO for another 30 days','Two days before his home was to be sold at a foreclosure sale','A TRO that is extended beyond the 14-day limit is treated as a preliminary injunction.'),
'mmlupro_1275':('without express permission of the mayor','After a few days of picketing, the school board seeks a temporary injunction','The ordinance gives the mayor unlimited discretion to permit or forbid speech, with no standards at all.'),
'mmlupro_1409':('which of the following is the WEAKEST defense for the city','For the past 20 years a city by the beach has been a popular location','So I need the argument that most clearly fails; the ordinance is in force and the dispute is live, so nothing has ended.'),
'mmlupro_1445':('a federal statute prohibits all commercial advertising of red meat products','Producers of red meat have challenged the statute as a violation of their free speech rights','A complete ban on truthful advertising of a lawful product goes further than is needed.'),
'mmlupro_1643':('the defendant was given a videotape made that day','The jury found in favor of the plaintiff and awarded her $5 million','The videotape did not exist at the time of the trial, so it is not evidence that could have been discovered then; it shows that her claim of permanent paralysis was false.'),
'mmlupro_1705':('the police noticed three women fidgeting rather nervously','The last ferry was getting ready to leave for the night','Nervous fidgeting is all the police had on this particular car before they searched it.'),
'mmlupro_2654':('he said he had dreamed he was falling to his death','When his wife asked him what had happened','He was frightened by the content of a dream and woke up; nothing says that he could not move.'),
'mmlupro_4811':('developments of the period 1450 to 1750 C.E.','Walter Rodney, historian, How Europe Underdeveloped Africa, 1974','So the development must have happened between 1450 and 1750, long before Europeans colonized the interior of the continent in the late 1800s.'),
'mmlupro_6110':('All patients enrolled in the study are informed that they would be treated with the tyrosine kinase inhibitor','Treatment efficacy is determined based on the results of complete blood counts and bone marrow assessments','So everyone knows which treatment is given: there is no blinding, and nothing says that the assignment was random.'),
'mmlupro_6562':('who has smoked 20 cigarettes daily for the past 38 years','a marked increase in shortness of breath while walking up a flight of stairs','She probably has chronic obstructive lung disease, so a drug that also blocks the beta-2 receptors of the airways would cause this.'),
'mmlupro_6614':('weighs 100 kg (220 lb); BMI is 37 kg/m2','works as a sales representative for a computer software company','She is obese, and with a normal examination and no red flags the excess weight is what should be addressed.'),
'mmlupro_6800':('no transillumination over the left maxillary sinus','Her husband and children have had no recent illness','This is acute bacterial maxillary sinusitis, so the question is which organism most commonly causes sinusitis.'),
'mmlupro_6829':('the world price of copper exceeds the domestic (U.S.) price of copper',None,'So U.S. producers can sell their copper abroad for more than they get at home.'),
}
FILL='the options one more time before I decide which of them is the best answer here'
ntok=lambda s:len(tok.encode(s,add_special_tokens=False))
TPL='Wait, let me re-read the question. It says: "%s".'
TPLF='Wait, let me think about this again. I should go through %s.'
PROBE='So the answer is ('
def base(r):
    T=trace_text(r); raw=r['raw']; raw=raw[7:] if raw.startswith('<think>') else raw
    assert T.endswith(raw); return T,len(T)-len(raw),raw
LAB=re.compile(r'^\s*(?:[-*]\s*)?(?:\*\*)?(?:Option\s+)?\(?([A-J])\)?(?:\*\*)?[\):.]')
def cuts(r):
    """early = paragraph break before the first paragraph judging a single option; late = break before the paragraph that first commits to the final letter"""
    T,off,raw=base(r); think=raw.split('</think>')[0]
    br=[mm.end() for mm in re.finditer(r'\n\n',think)]
    starts=[0]+br
    early=None
    for s,e in zip(starts,starts[1:]+[len(think)]):
        p=think[s:e]
        if LAB.match(p) and len(set(LAB.match(l).group(1) for l in p.split('\n') if LAB.match(l)))<=2 and len(p)>=80 and s>0: early=s; break
    late=None
    if r['pred']:
        X=r['pred']
        pat=re.compile(r'(?:answer(?: is| would be| should be|:)?|choose|pick|select|go with|correct (?:option|answer|choice)(?: is|:)?)\s*(?:likely |probably )?(?:option )?\(?%s\)?(?![A-Za-z])'%X)
        mm=pat.search(think)
        pos=mm.start() if mm else len(think)
        c=[s for s in starts if s<=pos and s>0]
        late=c[-1] if c else None
    return T,off,think,early,late
if __name__=='__main__':
    QUANT={'business','chemistry','physics','math','engineering'}
    items=[]
    for r in rows:
        if r['id'] not in ANN: continue
        T,off,think,early,late=cuts(r)
        st='cut' if r['cut'] else ('right' if r['correct'] else 'wrong')
        items.append(dict(id=r['id'],sample=r['sample'],status=st,pred=r['pred'],gold=r['gold'],early=early,late=late,ne=ntok(T[:off+early]) if early else None,nl=ntok(T[:off+late]) if late else None,nt=r['tokens'],frac=(late/len(think)) if late else None))
    for it in sorted(items,key=lambda i:(i['id'],i['sample'])): print(it['id'],it['sample'],it['status'],it['pred'],it['gold'],'early tok',it['ne'],'late tok',it['nl'],'of',it['nt'],'frac %.2f'%(it['frac'] or -1))
    for q,(c,n,i) in ANN.items():
        assert c in Q[q]['question'], (q,c)
        assert n is None or n in Q[q]['question'], (q,n)
        print(q,'cue tok',ntok(TPL%c),'non tok',ntok(TPL%n) if n else None)
