# Hypothesis: (pass 1 request) at </think> the parent rarely starts its answer with "The answer is (X)"; teacher-forced
# "The answer is (X)." variants show whether " answer" and EOS follow; L23 MLP activations show quantisation-silent neurons.
import json, re
from wsp.editor import trace_text
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
ids=['1740','9274','8198','6829','6014','9251','4839','8192','6577','2476','1643','453','5822']
ids=['mmlupro_'+i for i in ids]
pat=re.compile(r'answer is \(?\*?\*?([A-J])\)?')
texts=[];meta=[]
for q in ids:
    r=B[q]; t=trace_text(r); texts.append(t); meta.append(dict(id=q,kind='natural'))
for q in ids:
    r=B[q]; t=trace_text(r); i=t.find('</think>')
    m=pat.findall(r['think']); X=m[-1] if m else 'D'
    texts.append(t[:i+len('</think>')]+f'The answer is ({X}).'); meta.append(dict(id=q,kind='variant',X=X))
req={"type":"read","model":"original","texts":texts,
     "return":{"token_logprobs":True,"topk":20,"mlp":[23],"inputs":["model.layers.23.feed_forward.w1"],"positions":"all"}}
json.dump(req,open('work/req_p1.json','w')); json.dump(meta,open('work/p1_meta.json','w'))
from wsp.editor import tok
print(sum(len(tok(t)['input_ids']) for t in texts), [m.get('X') for m in meta[13:]])
