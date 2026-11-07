# Hypothesis: the content starts right after the </think> token with a fixed '\n\n' token; check tokenization around </think> and the end of content.
import json
from wsp.editor import tok, render_think, trace_text
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
r=B[0]
t=trace_text(r)
ids=tok(t)['input_ids']
i=t.find('</think>')
pre=len(tok(t[:i])['input_ids'])
print('n tokens',len(ids),'close at',pre)
for j in range(pre-3,pre+6): print(j,ids[j],repr(tok.decode([ids[j]])))
print('last tokens',[ (x,tok.decode([x])) for x in ids[-6:]])
print(repr(t[-80:]))
print(repr(render_think(r['id'])[-200:]))
for w in ['The','The answer','\n\n','<|im_end|>','</think>','**','We','To']:
    print(repr(w), tok(w,add_special_tokens=False)['input_ids'])
print(tok.eos_token, tok.eos_token_id)
