# Hypothesis: at </think>, '"The'(after it) and ')".'(after it) the parent's own next-token preferences are measurable; check base log-probs of the intended tokens and the competitors that a shared write would boost.
import json, numpy as np
from wsp.editor import *
d=json.load(open('work/p2_texts.json')); D='work/fwd_02'
meta=json.load(open(f'{D}/meta.json')); off=np.load(f'{D}/offsets.npy'); lp=np.load(f'{D}/token_logprobs.npy')
ti=np.load(f'{D}/topk_ids.npy'); tv=np.load(f'{D}/topk_logprobs.npy')
# token_logprobs per text has n-1 entries; build global index of lp: text i entries start at off[i]-i
TH,QT,ANS,PE,IM=124902,14225,5141,94078,124900
pos={'think':[],'qthe':[],'pe':[]}; rows=[]
for i,t in enumerate(d['texts']):
    ids=tok(t,add_special_tokens=False).input_ids; assert len(ids)==meta['n_tokens'][i]
    if d['kind'][i]=='cut': continue
    p=ids.index(TH); assert ids[p+1]==QT and ids[p+6]==PE and ids[p+7]==IM, [tok.decode([x]) for x in ids[p:p+8]]
    g=off[i]
    base=off[i]-i
    r=dict(id=d['ids'][i],kind=d['kind'][i],lp_qthe=lp[base+p],lp_ans=lp[base+p+1],lp_is=lp[base+p+2],lp_paren=lp[base+p+3],lp_letter=lp[base+p+4],lp_pe=lp[base+p+5],lp_im=lp[base+p+6])
    r['top_think']=[ (tok.decode([int(a)]).encode('ascii','replace').decode(),round(float(b),2)) for a,b in zip(ti[g+p][:4],tv[g+p][:4])]
    r['top_qthe']=[ (tok.decode([int(a)]).encode('ascii','replace').decode(),round(float(b),2)) for a,b in zip(ti[g+p+1][:4],tv[g+p+1][:4])]
    r['top_pe']=[ (tok.decode([int(a)]).encode('ascii','replace').decode(),round(float(b),2)) for a,b in zip(ti[g+p+6][:4],tv[g+p+6][:4])]
    pos['think'].append(g+p); pos['qthe'].append(g+p+1); pos['pe'].append(g+p+6); rows.append(r)
for k in ['lp_qthe','lp_ans','lp_is','lp_paren','lp_letter','lp_pe','lp_im']:
    v=np.array([r[k] for r in rows]); print(k,'median',round(float(np.median(v)),3),'min',round(float(v.min()),3))
for r in rows[:6]: print(r['id'],r['top_think'],'|',r['top_qthe'],'|',r['top_pe'])
json.dump({k:[int(x) for x in v] for k,v in pos.items()},open('work/p2_targetpos.json','w'))
