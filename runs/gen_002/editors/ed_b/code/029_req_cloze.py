# Hypothesis: the model holds answer knowledge that the letter format does not use: scoring each option's text likelihood after the question ("cloze") picks gold above chance on questions the MC protocol gets wrong.
from wsp.editor import *
import json, numpy as np
rows=[json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')]; qs=questions()
texts=[];meta=[];tot=0
rng=np.random.default_rng(0); order=rng.permutation(len(rows))
for i in order:
    r=rows[i]; q=qs[r['id']]; s=render(r['id'])
    cut=s.index('\n\n(A)'); pre=s[:cut]
    tail='\n\nAnswer with the correct answer.<|im_end|>\n<|im_start|>assistant\n<think></think>The answer is: '
    base=pre+tail; nb=len(tok.encode(base))
    tt=[base+o for o in q['options']]; nt=sum(len(tok.encode(t)) for t in tt)
    if tot+nt>190000 or len(texts)+len(tt)>1024: continue
    if nb>250: continue
    for L,t in enumerate(tt): texts.append(t); meta.append((r['id'],L,nb))
    tot+=nt
assert s.startswith(pre)
print(len(texts), tot, len(set(m[0] for m in meta)))
json.dump(meta,open('work/cloze_meta.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"token_logprobs":True}},open('work/req_cloze.json','w'))
print(repr(texts[0][-300:]))
