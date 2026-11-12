# Analyse pass 1: (a) log-prob of the sampled letter at in-thinking commitment phrases; (b) rate of low-probability sampled
# tokens (p<0.05) in right vs wrong traces of the same mixed questions; (c) the words at low-prob tokens.
import json, numpy as np, re, collections
from wsp.editor import *
idx=json.load(open('work/read1_index.json')); req=json.load(open('work/req_read1.json'))
meta=json.load(open('work/fwd_01/meta.json')); lp=np.load('work/fwd_01/token_logprobs.npy')
off=0; res=[]; lowwords=collections.Counter(); lowwords_w=collections.Counter()
for m,text,n in zip(idx,req['texts'],meta['n_tokens']):
    ids=tok(text,add_special_tokens=False).input_ids; assert len(ids)==n, (len(ids),n)
    l=lp[off:off+n-1]; off+=n-1
    pl=len(tok(render_think(m['id']),add_special_tokens=False).input_ids)
    gl=l[pl-1:]   # log-probs of generated tokens
    toks=[tok.decode([i]) for i in ids[pl:]]
    # commitment letters: token is a single letter A-J preceded by "(" and text before contains go with/answer is/choose
    com=[]
    for j in range(2,len(toks)):
        if toks[j].strip() in list("ABCDEFGHIJ") and toks[j-1].endswith("("):
            ctx="".join(toks[max(0,j-8):j]).lower()
            if re.search(r"(answer|go with|choose|pick|select)",ctx): com.append((j,toks[j].strip(),float(np.exp(gl[j]))))
    low=np.where(gl<np.log(0.05))[0]
    for j in low:
        w=toks[j].strip()
        (lowwords if m['correct'] else lowwords_w)[w]+=1
    res.append({**m,'n_gen':len(toks),'low_rate':len(low)/len(toks),'com':com})
for k in (1,0):
    sub=[r for r in res if r['correct']==k]
    print('correct' if k else 'wrong', len(sub), 'low-prob(<.05) tokens per 1k:', round(1000*np.mean([r['low_rate'] for r in sub]),1))
    ps=[c[2] for r in sub for c in r['com']]
    print('   commitment letters:',len(ps),'median p',round(float(np.median(ps)),3) if ps else None,'share p<0.9',round(np.mean([p<0.9 for p in ps]),2) if ps else None)
print('first commitment letter p (wrong):',[ (r['id'][-5:],r['com'][0][1],round(r['com'][0][2],2)) for r in res if not r['correct'] and r['com']])
print('top low-prob words right:',lowwords.most_common(25))
print('top low-prob words wrong:',lowwords_w.most_common(25))
