# Hypothesis: at a two-finalist statement the model prefers the FIRST-MENTIONED finalist because of mention order (not content); test by swapping the two letters in the pair statement and forcing a commit right after the statement sentence.
import json, re
from wsp.editor import *
pat=re.compile(r"(?:between|either)\s*\(?([A-J])\)?\s*(?:,|and|or)\s*\(?([A-J])\)?(?![A-Za-z])|\(?\b([A-J])\)?\s+(?:or|vs\.?|versus)\s+\(?([A-J])\)?(?![A-Za-z])")
items=json.load(open('work/probe_items_all.json'))
SUF="\n\nThus I'll go with ("
out=[]
for it in items:
    t=it['text']; ms=list(pat.finditer(t)); m=ms[-1]
    gi=(1,2) if m.group(1) else (3,4)
    s1,e1=m.span(gi[0]); s2,e2=m.span(gi[1])
    # end of the sentence containing the pair
    j=m.end(); 
    while j<len(t) and t[j] not in '.?!\n': j+=1
    if j<len(t) and t[j] in '.?!': j+=1
    base=t[:j]
    sw=base[:s1]+base[s2:e2]+base[e1:s2]+base[s1:e1]+base[e2:]
    for kind,txt in (('orig',base),('swap',sw)):
        x=txt+SUF; n=len(tok.encode(x,add_special_tokens=False))
        out.append(dict(it,kind=kind,text=x,ntok=n,first=t[s1:e1],second=t[s2:e2],tail=t[max(0,m.start()-80):j][-160:]))
import re as _r
kw=_r.compile(r'answer|either|between|option|choose|pick|lean|decide',_r.I)
pairs=[(out[i],out[i+1]) for i in range(0,len(out),2) if out[i]['ntok']<=4096 and kw.search(out[i]['tail'][-120:])]
pairs.sort(key=lambda p:p[0]['ntok'])
sel=[];tot=0
for a,b in pairs:
    if tot+a['ntok']+b['ntok']<=198000: sel+= [a,b]; tot+=a['ntok']+b['ntok']
print('pairs',len(sel)//2,'tokens',tot)
for s in sel[:6:2]: print(repr(s['tail'][-120:].encode('ascii','replace').decode()))
json.dump(sel,open('work/swap_sel.json','w'))
json.dump({'type':'read','model':'original','texts':[s['text'] for s in sel],'return':{'topk':50,'positions':'last'}},open('work/req_p4_read.json','w'))
