# Hypothesis: out-of-context cloze (option text as the assistant's answer to the bare question) carries gold information
# on questions the letter protocol gets wrong (gen 2: 25% on 72 q). Build the read request on ~100 DEV questions of pass 1.
import json
from wsp.editor import *
Q = questions()
ids = json.load(open('work/pass1_ids.json'))[256:]
texts, meta, tot = [], [], 0
for q in ids:
    s = render(q)
    a = s.rfind('\n\n(A) ', 0, s.rfind('\n\nAnswer with the letter'))
    head = s[:a]
    tail = '<|im_end|>\n<|im_start|>assistant\n<think></think>The answer is '
    opts = Q[q]['options']
    new = [head + tail + o for o in opts]
    n = sum(len(tok.encode(t, add_special_tokens=False)) for t in new)
    if len(texts) + len(new) > 1024 or tot + n > 195000: break
    for k, t in enumerate(new):
        texts.append(t); meta.append([q, k, len(tok.encode(head + tail, add_special_tokens=False))])
    tot += n
print(len(texts), 'texts', len(set(m[0] for m in meta)), 'questions', tot, 'tokens')
assert texts[0].endswith(Q[meta[0][0]]['options'][0])
json.dump(meta, open('work/cloze_meta.json', 'w'))
json.dump({'type': 'read', 'model': 'original', 'texts': texts, 'return': {'token_logprobs': True}}, open('work/req_cloze.json', 'w'))
