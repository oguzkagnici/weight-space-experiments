# Hypothesis: right samples have the decisive cue (or stem content generally) closer to the first commitment than wrong samples (inside the 256-token window of the sliding attention layers).
import json, sys, re, importlib, collections, numpy as np
sys.path.insert(0,'code'); m=importlib.import_module('001_overview'); b=importlib.import_module('005_build_ins'); k9={}
src=open('code/009_cue_mentions.py').read(); exec(src[src.index('KEY='):src.index('for q,pat')],k9); KEY=k9['KEY']
f3=importlib.import_module('003_stem_restate')
from wsp.editor import *
Q=questions(); rows=m.load_rows(); QUANT={'business','chemistry','physics','math','engineering'}
ntok=b.ntok
out=[]
for r in rows:
    if r['cut'] or not r['pred']: continue
    T,off,think,early,late=b.cuts(r)
    X=r['pred']; pat=re.compile(r'(?:answer(?: is| would be| should be|:)?|choose|pick|select|go with|correct (?:option|answer|choice)(?: is|:)?)\s*(?:likely |probably )?(?:option )?\(?%s\)?(?![A-Za-z])'%X)
    mm=pat.search(think); pos=mm.start() if mm else len(think)
    pre=think[:pos]
    # window = last ~256 tokens before the commitment (approximate by characters: walk back)
    ids=tok(pre,add_special_tokens=False,return_offsets_mapping=True); om=ids['offset_mapping']
    ws=om[-256][0] if len(om)>256 else 0
    win=pre[ws:]; q=Q[r['id']]; sw=f3.words(q['question']); ow=set(w for o in q['options'] for w in f3.words(str(o)))
    so=set(w for w in sw if w not in f3.STOP and w not in ow and len(w)>2); ww=f3.words(win); s3=set(zip(sw,sw[1:],sw[2:]))
    d=dict(id=r['id'],sample=r['sample'],correct=r['correct'],verbal=r['category'] not in QUANT,so=1000*sum(w in so for w in ww)/max(1,len(ww)),g3=1000*sum(g in s3 for g in zip(ww,ww[1:],ww[2:]))/max(1,len(ww)),stemlen=len(sw))
    if r['id'] in KEY:
        ms=[x.start() for x in re.finditer(KEY[r['id']],pre)]
        d['cue_in_win']=float(any(x>=ws for x in ms)); d['cue_cnt_win']=sum(x>=ws for x in ms); d['dist']=len(om)-next((i for i,(a,c) in enumerate(om) if a>=ms[-1]),len(om)) if ms else 9999
    out.append(d)
print('cue questions with right samples: cue key word within the last 256 tokens before the first commitment; distance (tokens) from the last cue mention to the commitment')
for q in ('mmlupro_6110','mmlupro_2654','mmlupro_6562','mmlupro_6800'):
    for d in sorted([d for d in out if d['id']==q],key=lambda d:d['sample']): print('  ',q,d['sample'],'RIGHT' if d['correct'] else 'wrong','in window:',d['cue_in_win'],'count',d['cue_cnt_win'],'distance',d['dist'])
for sel,tag in ((lambda d:d['verbal'],'verbal mixed'),(lambda d:d['verbal'] and d['stemlen']>=40,'verbal mixed, stem >= 40 words')):
    byq=collections.defaultdict(lambda:{0:[],1:[]})
    for d in out:
        if sel(d): byq[d['id']][d['correct']].append(d)
    mx={q:v for q,v in byq.items() if v[0] and v[1]}
    for k in ('so','g3'):
        dd=np.array([np.mean([x[k] for x in v[1]])-np.mean([x[k] for x in v[0]]) for v in mx.values()])
        print('%s (%d questions): %s in the 256 tokens before the first commitment: right %.1f wrong %.1f, within-question diff %+.1f (SE %.1f), right>wrong %d/%d'%(tag,len(mx),k,np.mean([np.mean([x[k] for x in v[1]]) for v in mx.values()]),np.mean([np.mean([x[k] for x in v[0]]) for v in mx.values()]),dd.mean(),dd.std(ddof=1)/np.sqrt(len(dd)),(dd>0).sum(),len(dd)))
