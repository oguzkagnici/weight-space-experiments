# Hypothesis: (pass 2 request) some opener of the visible answer leads to "answer is (X)" and then <|im_end|> with high natural
# probability, so fewer/lighter pushes are needed; L22/L23 MLP inputs along these texts give gate data for any opener position.
import json, re
from wsp.editor import trace_text, tok
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
ids=['mmlupro_'+i for i in ['1740','9274','8198','6829','6014','9251','4839','8192','6577','2476']]
pat=re.compile(r'answer is \(?\*?\*?([A-J])\)?')
OPEN={'O2':'"The answer is ({X})".','O4':'Therefore, the answer is ({X}).','O5':'The correct answer is ({X}).',
      'O8':'**The answer is ({X}).**','O9':'Thus, the answer is ({X}).','O10':'The final answer is ({X}).'}
texts=[];meta=[]
for o,s in OPEN.items():
    for q in ids:
        r=B[q]; t=trace_text(r); i=t.find('</think>')
        m=pat.findall(r['think']); X=m[-1] if m else 'D'
        texts.append(t[:i+len('</think>')]+s.format(X=X)); meta.append(dict(id=q,open=o,X=X))
req={"type":"read","model":"original","texts":texts,
     "return":{"token_logprobs":True,"topk":20,"inputs":["model.layers.22.feed_forward.w1","model.layers.23.feed_forward.w1"],"positions":"all"}}
json.dump(req,open('work/req_p2.json','w')); json.dump(meta,open('work/p2_meta.json','w'))
print(len(texts), sum(len(tok(t,add_special_tokens=False)['input_ids']) for t in texts))
for o,s in OPEN.items(): print(o, [tok.decode([x]) for x in tok(s.format(X='D'),add_special_tokens=False)['input_ids']])
