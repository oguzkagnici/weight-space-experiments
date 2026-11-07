# Hypothesis: (pass 3 request) the edited model writes '"The', ' answer', <|im_end|> at the three post-think positions with P~1, changes thinking-token log-probs only by the tiny fold residual, and its gates never fire on unseen thinking traces (detectable as top-1 = a target token where the text differs).
import json, numpy as np
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
d2=json.load(open('work/p2_texts.json')); used=set(d2['ids'])|set(json.load(open('work/p1_texts.json'))['ids'])
texts=list(d2['texts']); kinds=list(d2['kind']); ids=list(d2['ids']); tot=sum(d2['n'])
for r in B:
    if r['id'] in used: continue
    t=trace_text(r); n=len(tok(t,add_special_tokens=False).input_ids)
    if tot+n>195000: continue
    texts.append(t); kinds.append('oos'); ids.append(r['id']); tot+=n
print(len(texts),tot,kinds.count('oos'))
json.dump({'ids':ids,'kind':kinds,'texts':texts},open('work/p3_texts.json','w'))
json.dump({'type':'read','model':'edited','texts':texts,'return':{'token_logprobs':True,'topk':5,'positions':'all'}},open('work/req3.json','w'))
