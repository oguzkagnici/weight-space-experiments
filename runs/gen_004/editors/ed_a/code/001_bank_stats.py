# Hypothesis: truncated traces spend budget on reconsideration episodes that start with sentence-initial markers (Wait/Alternatively/But maybe/Let's double-check); count them and the position of first in-think answer statement.
import json, re, numpy as np
from wsp.editor import tok
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl')]
markers=['Wait','Alternatively','But maybe','Maybe','Hmm','Actually','But','Let\'s double-check','Let\'s verify','Let\'s check','Double-check','Check','However','Let\'s compute','Let\'s re','But we','So answer','Thus answer','So the answer','Thus the answer','Option']
def stats(rs):
    out={}
    for m in markers:
        c=0;T=0
        for r in rs:
            t=r['think']; ntok=int(r['tokens'])
            # sentence-initial: preceded by newline or ". "
            c+=len(re.findall(r'(?:^|[\.\n!?]\s*)'+re.escape(m)+r'\b',t))
            T+=ntok
        out[m]=1000*c/T
    return out
cut=[r for r in rows if r['cut']==1]; fin=[r for r in rows if r['cut']==0]
sc,sf=stats(cut),stats(fin)
for m in markers: print(f'{m:20s} cut {sc[m]:5.2f}  fin {sf[m]:5.2f}')
# lengths and answer statements
for name,rs in [('cut',cut),('fin',fin)]:
    nl=[r['think'].count('\n\n') for r in rs]
    print(name,len(rs),'mean paragraphs',np.mean(nl),'chars/tok',np.mean([len(r['think'])/int(r['tokens']) for r in rs]))
# first "answer is" in think
ans=re.compile(r'answer (?:is|should be|would be)[^\n]{0,20}?\(?([A-J])\)?')
for r in cut:
    t=r['think']; m=list(ans.finditer(t))
    if m:
        f=m[0].start()/len(t)
        letters=[x.group(1) for x in m]
        print(r['id'],r['category'],'gold',r['gold'],'first ans at %.2f'%f,'letters',''.join(letters[:12]), 'n',len(m))
