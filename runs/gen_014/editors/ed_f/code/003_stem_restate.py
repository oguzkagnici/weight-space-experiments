# Hypothesis (step 1a): right samples re-state stem content while judging options more than wrong samples of the same question.
import json, sys, re, importlib, collections, numpy as np
sys.path.insert(0,'code'); m=importlib.import_module('001_overview')
from wsp.editor import *
Q=questions()
STOP=set("a an the of to in on at for and or but is are was were be been being this that these those it its as by with from which what who whom whose when where why how not no nor do does did has have had will would can could should may might must shall if then than so such there their they them he she his her him you your we our us i me my following most best likely each all any some one two three more less also into over under between about after before during per out up only very same other another both either neither while because due".split())
W=re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?")
words=lambda s:[w.lower() for w in W.findall(s)]
def features(r):
    q=Q[r['id']]; stem=q['question']; sw=words(stem); ow=set(w for o in q['options'] for w in words(str(o)))
    stem_only=set(w for w in sw if w not in STOP and w not in ow and len(w)>2)
    s3=set(zip(sw,sw[1:],sw[2:])); s4=set(zip(sw,sw[1:],sw[2:],sw[3:]))
    raw=r['raw']; raw=raw[7:] if raw.startswith('<think>') else raw
    think=raw.split('</think>')[0]
    paras=[p for p in think.split('\n\n') if p.strip()]
    lab=re.compile(r'^\s*(?:[-*]\s*)?(?:\*\*)?\(?(?:Option\s+)?\(?([A-J])\)?[\):.]')
    isj=[bool(lab.match(p)) or bool(re.search(r'\([A-J]\)|[Oo]ption [A-J]\b',p)) for p in paras]
    if not any(isj): return None
    first=isj.index(True)
    # skip a pure option-list restatement: judging starts at the first labelled paragraph AFTER the first paragraph group that lists >=6 options without verdict words
    J=[p for p,j in zip(paras[first:],isj[first:])]           # judging region = everything from the first option mention
    Jj=[p for p,j in zip(paras[first:],isj[first:]) if j]      # paragraphs that mention an option
    txt=' '.join(J); jw=words(txt); n=max(1,len(jw))
    g3=list(zip(jw,jw[1:],jw[2:])); g4=list(zip(jw,jw[1:],jw[2:],jw[3:]))
    quotes=re.findall(r'"([^"\n]{6,200})"',txt)
    def inq(qs):
        w=words(qs); 
        if len(w)<2: return False
        g=list(zip(w,w[1:])); 
        return sum((a,b) in set(zip(sw,sw[1:])) for a,b in g)>=max(1,0.6*len(g))
    nq=sum(inq(x) for x in quotes)
    mark=len(re.findall(r'\b(?:the question (?:says|states|asks|mentions|is|specifies|describes)|the (?:stem|scenario|passage|vignette|case|problem statement|prompt)\b|re-?read|read(?:ing)? the question|according to the (?:question|stem|scenario|passage)|as (?:stated|given|described|mentioned)|the phrase|it says|they say|they mention|the key (?:phrase|word|clue|detail))',txt,re.I))
    pj=[ (any(g in s3 for g in zip(words(p),words(p)[1:],words(p)[2:])) ) for p in Jj]
    return dict(n=n,nJ=len(Jj),so=sum(w in stem_only for w in jw)/n*1000, g3=sum(g in s3 for g in g3)/n*1000, g4=sum(g in s4 for g in g4)/n*1000,
                quote=nq/n*1000, nquote=nq, mark=mark/n*1000, pj=float(np.mean(pj)) if pj else np.nan, stemlen=len(sw), first=first/len(paras), anyq=float(nq>0))
if __name__=='__main__':
    rows=m.load_rows(); QUANT={'business','chemistry','physics','math','engineering'}
    R=[]
    for r in rows:
        if r['cut']: continue
        f=features(r)
        if f is None: continue
        f.update(id=r['id'],sample=r['sample'],cat=r['category'],correct=r['correct'],verbal=r['category'] not in QUANT); R.append(f)
    keys=['so','g3','g4','quote','anyq','mark','pj','n']
    print('finished traces with judging region',len(R))
    def cmpq(tag,sel):
        byq=collections.defaultdict(lambda:{0:[],1:[]})
        for f in R:
            if sel(f): byq[f['id']][f['correct']].append(f)
        mixed={q:v for q,v in byq.items() if v[0] and v[1]}
        print('\n%s: mixed questions %d (right samples %d, wrong %d)'%(tag,len(mixed),sum(len(v[1]) for v in mixed.values()),sum(len(v[0]) for v in mixed.values())))
        for k in keys:
            d=np.array([np.nanmean([f[k] for f in v[1]])-np.nanmean([f[k] for f in v[0]]) for v in mixed.values()]); d=d[~np.isnan(d)]
            mr=np.nanmean([np.nanmean([f[k] for f in v[1]]) for v in mixed.values()]); mw=np.nanmean([np.nanmean([f[k] for f in v[0]]) for v in mixed.values()])
            print('  %-6s right %8.2f wrong %8.2f | within-question diff %+8.2f (SE %.2f) | right>wrong in %d/%d'%(k,mr,mw,d.mean(),d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),len(d)))
        return mixed
    cmpq('ALL mixed',lambda f:True); cmpq('VERBAL mixed',lambda f:f['verbal']); cmpq('VERBAL mixed, stem >= 40 words',lambda f:f['verbal'] and f['stemlen']>=40)
    # between questions: verbal finished traces, right vs wrong (question means), with stem length control
    print('\nBetween traces (verbal, finished): right vs wrong, all questions')
    for long in (0,1):
        A=[f for f in R if f['verbal'] and (f['stemlen']>=40)==bool(long)]
        qm=collections.defaultdict(list)
        for f in A: qm[(f['id'],f['correct'])].append(f)
        for k in keys:
            r_=[np.nanmean([f[k] for f in v]) for (q,c),v in qm.items() if c==1]; w_=[np.nanmean([f[k] for f in v]) for (q,c),v in qm.items() if c==0]
            print('  stem%s40w %-6s right %8.2f (n %d) wrong %8.2f (n %d) diff %+.2f (SE %.2f)'%('>=' if long else '<',k,np.nanmean(r_),len(r_),np.nanmean(w_),len(w_),np.nanmean(r_)-np.nanmean(w_),np.sqrt(np.nanvar(r_,ddof=1)/len(r_)+np.nanvar(w_,ddof=1)/len(w_))))
    json.dump(R,open('work/stem_feats.json','w'))
