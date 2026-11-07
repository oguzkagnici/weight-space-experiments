# Hypothesis: teacher-forcing prompt+think+'</think>"The answer is (X)".<|im_end|>' (X = think's last stated letter) on the parent gives the exact L23 inputs the edited model will see at </think>, '"The' and ')".' (an L23 edit only changes logits at its own position).
import json, re, numpy as np
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
pat=re.compile(r'answer is \(([A-J])\)')
syn=[];ids=[];kind=[]
for r in B:
    closed='</think>' in r['raw']
    if not closed: continue
    m=pat.findall(r['think'])
    if not m: continue
    full=trace_text(r); k=full.find('</think>')
    t=full[:k]+'</think>"The answer is ('+m[-1]+')".<|im_end|>'
    syn.append(t); ids.append(r['id']); kind.append('syn_cut' if r['cut'] else 'syn_fin')
n=[len(tok(t,add_special_tokens=False).input_ids) for t in syn]
print(len(syn), sum(n))
# fill with cut, non-closed traces (thinking negatives)
p1=set(json.load(open('work/p1_texts.json'))['ids'])
rng=np.random.default_rng(3)
cuts=[r for r in B if r['cut'] and '</think>' not in r['raw'] and r['id'] not in p1]
rng.shuffle(cuts); tot=sum(n)
for r in cuts:
    t=trace_text(r); k=len(tok(t,add_special_tokens=False).input_ids)
    if tot+k>96000: break
    syn.append(t); ids.append(r['id']); kind.append('cut'); n.append(k); tot+=k
print(len(syn), tot, kind.count('cut'))
json.dump({'ids':ids,'kind':kind,'n':n,'texts':syn},open('work/p2_texts.json','w'))
req={'type':'read','model':'original','texts':syn,'return':{'token_logprobs':True,'topk':10,'mlp':[23],'inputs':['model.layers.23.feed_forward.w1'],'positions':'all'}}
json.dump(req,open('work/req2.json','w'))
