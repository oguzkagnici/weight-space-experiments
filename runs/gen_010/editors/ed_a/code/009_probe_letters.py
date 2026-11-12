# Hypothesis: at forced commit after a two-finalist statement, the letter logits carry an early-letter prior that decides near-ties (gold-later pairs lost); measure P(a),P(b) and the letter prior via logit lens.
import json, numpy as np, glob, os, sys, torch
from wsp.editor import *
d=sys.argv[1]
meta=json.load(open(d+'/meta.json')); print({k:v for k,v in meta.items() if k not in ('n_tokens','files')}); print(list(meta['files'].keys())[:40])
sel=json.load(open('work/probe_sel.json'))
L='ABCDEFGHIJ'
ti=np.load(d+'/topk_ids.npy'); tl=np.load(d+'/topk_logprobs.npy'); print(ti.shape)
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy(); g=load('model.embedding_norm.weight').numpy()
h=np.load(glob.glob(d+'/hidden_states_24*.npy')[0]) if glob.glob(d+'/hidden_states_24*.npy') else None
print('h24', None if h is None else h.shape)
hn=h/np.sqrt((h**2).mean(1,keepdims=True)+1e-5)*g
LL=hn@E.T   # logit-lens letter logits (exact final logits if h24 is pre-final-norm)
np.save('work/p1_letter_logits.npy',LL)
rows=[]
for k,it in enumerate(sel):
    lp={int(i):float(v) for i,v in zip(ti[k],tl[k])}
    P=np.array([np.exp(lp.get(LETTER_IDS[j],-30)) for j in range(10)])
    a,b=sorted([it['a'],it['b']]); ia,ib=L.index(a),L.index(b)
    rows.append(dict(id=it['id'],s=it['sample'],gold=it['gold'],pred=it['pred'],cut=it['cut'],a=a,b=b,Pa=P[ia],Pb=P[ib],Pl=P.sum(),
        top=L[int(np.argmax(P))], dl=LL[k,ia]-LL[k,ib]))
ok=[r for r in rows if r['gold'] in (r['a'],r['b'])]
print('letter mass mean',np.mean([r['Pl'] for r in rows]).round(3), 'P(a)+P(b) mean', np.mean([r['Pa']+r['Pb'] for r in rows]).round(3))
for name,grp in [('gold earlier',[r for r in ok if r['gold']==r['a']]),('gold later',[r for r in ok if r['gold']==r['b']])]:
    print(name,len(grp),'probe picks gold',sum((r['Pa']>r['Pb'])==(r['gold']==r['a']) for r in grp),
          'mean P(gold)',np.mean([r['Pa'] if r['gold']==r['a'] else r['Pb'] for r in grp]).round(2),
          'trace right',sum(r['pred']==r['gold'] for r in grp))
print('mean logit(earlier)-logit(later) all pairs', np.mean([r['dl'] for r in rows]).round(2), 'P(earlier>later)', np.mean([r['Pa']>r['Pb'] for r in rows]).round(2))
for r in rows: print(r['id'],r['s'],'gold',r['gold'],'pred',r['pred'],'cut',r['cut'],'pair',r['a']+r['b'],'Pa %.2f Pb %.2f'%(r['Pa'],r['Pb']),'dl %.2f'%r['dl'])
json.dump(rows,open('work/p1_rows.json','w'))
