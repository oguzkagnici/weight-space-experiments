# Pass 2: latent preference (forced "So the answer is (" after the last finalist statement) vs the realised final pick.
import json, numpy as np
from wsp.editor import *
items=json.load(open('work/probe2_index.json'))
ids=np.load('work/fwd_02/topk_ids.npy'); lps=np.load('work/fwd_02/topk_logprobs.npy')
L="ABCDEFGHIJ"; lid={i:LETTER_IDS[k] for k,i in enumerate(L)}
rows=[]
for it,ti,tl in zip(items,ids,lps):
    pr={l:0.0 for l in L}
    for a,b in zip(ti,tl):
        for l in L:
            if a==lid[l]: pr[l]+=float(np.exp(b))
    s=sum(pr.values()); top=max(pr,key=pr.get)
    pair=it['pair']; pp={l:pr[l] for l in pair}; ptop=max(pp,key=pp.get)
    rows.append({**it,'letter_mass':s,'probe_top':top,'probe_pair_top':ptop,'p_gold':pr[it['gold']]/max(s,1e-9)})
fin=[r for r in rows if not r['cut']]
print('n',len(rows),'finished',len(fin),'median letter mass',round(float(np.median([r['letter_mass'] for r in rows])),3))
print('probe top == gold:',sum(r['probe_top']==r['gold'] for r in fin),'| realised correct:',sum(r['correct'] for r in fin),'of',len(fin))
g=[r for r in fin if r['gold'] in r['pair']]
print('gold in pair:',len(g),'probe picks gold (within pair):',sum(r['probe_pair_top']==r['gold'] for r in g),
      '| realised picks gold:',sum(r['correct'] for r in g))
print('probe top == realised final pred:',sum(r['probe_top']==r['pred'] for r in fin),'of',len(fin))
for r in g: print(r['id'][-6:],r['sample'],'pair',''.join(r['pair']),'gold',r['gold'],'pred',r['pred'],'probe',r['probe_top'],'p_gold',round(r['p_gold'],2),'frac',r['frac'])
