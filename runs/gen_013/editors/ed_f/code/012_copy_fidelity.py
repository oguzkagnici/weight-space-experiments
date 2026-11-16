# Hypothesis: numbers copied from the question into the reasoning are low-probability at some positions (natural mis-copy risk); measure P(first token of a copied number) in clean pass-1 texts.
import json, re, numpy as np
from wsp.editor import *
D='work/fwd_01'; meta=json.load(open('work/p1_meta.json')); M=json.load(open(D+'/meta.json')); T=json.load(open('work/p1_texts.json'))
tl=np.load(D+'/token_logprobs.npy'); off=np.cumsum([0]+[x-1 for x in M['n_tokens']])
Q=questions(); NUM=re.compile(r'(?<![\w.,])\d+(?:,\d{3})*(?:\.\d+)?(?![\w]|[.,]\d)')
seen=set(); lps=[]; lps_other=[]
for n,m in enumerate(meta):
    if m['kind']!='clean': continue
    text=T[n]; enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); offs=enc['offset_mapping']
    assert len(offs)==M['n_tokens'][n],(len(offs),M['n_tokens'][n])
    st=text.index('<think>')+7; q=Q[m['id']]; qtxt=q['question']+' '+' '.join(q['options'])
    qnums={x.group(0) for x in NUM.finditer(qtxt)}
    starts={o[0]:i for i,o in enumerate(offs)}
    for mm in NUM.finditer(text,st):
        key=(m['id'],m['sample'],mm.start(),text[max(st,mm.start()-40):mm.end()])
        if key in seen or mm.start() not in starts: continue
        seen.add(key); i=starts[mm.start()]
        if i==0: continue
        lp=tl[off[n]+i-1]
        (lps if (mm.group(0) in qnums and len(mm.group(0))>=2) else lps_other).append((lp,mm.group(0),text[max(st,mm.start()-60):mm.end()+10]))
for name,L in (('numbers that occur in the question/options (copies)',lps),('other numbers',lps_other)):
    v=np.array([x[0] for x in L]); p=np.exp(v)
    # probability of sampling this token at T=0.6 cannot be computed without the full distribution; report P and a two-way approximation p^(1/.6)/(p^(1/.6)+(1-p)^(1/.6))
    pt=p**(1/.6)/(p**(1/.6)+(1-p)**(1/.6))
    print('%s: n %d | median P %.3f | share P<.5 %.3f, P<.9 %.3f | mean P %.3f | approx. mean P at T=0.6 %.3f'%(name,len(v),np.median(p),(p<.5).mean(),(p<.9).mean(),p.mean(),pt.mean()))
low=sorted(lps)[:12]
for lp,s,ctx in low: print('  P %.3f | %s | %r'%(np.exp(lp),s,ctx.encode('ascii','replace').decode()))
