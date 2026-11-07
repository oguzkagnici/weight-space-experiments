# Hypothesis: (a) at </think> the parent seldom starts with "The answer"; after a forced "The answer is (X)." EOS is likely or not;
# (b) some L23 neurons are Q8_0-silent (quantised activation 0) at every prompt+thinking token; (c) the </think>/first-content
# positions are linearly separable from all thinking tokens at the L23 MLP input.
import json, numpy as np
from wsp.editor import tok
D='work/fwd_01/'
M=json.load(open(D+'meta.json')); meta=json.load(open('work/p1_meta.json'))
req=json.load(open('work/req_p1.json')); texts=req['texts']
off=np.load(D+'offsets.npy'); n=M['n_tokens']
TI=np.load(D+'topk_ids.npy'); TV=np.load(D+'topk_logprobs.npy')
CLOSE=124902; EOS=124900
ids_all=[tok(t,add_special_tokens=False)['input_ids'] for t in texts]
assert [len(x) for x in ids_all]==n
def top(pos,k=6):
    return ' '.join(f"{tok.decode([int(i)])!r}:{np.exp(v):.2f}" for i,v in zip(TI[pos][:k],TV[pos][:k]))
cpos=[]
for j,ids in enumerate(ids_all):
    c=ids.index(CLOSE); cpos.append(c)
json.dump(cpos,open('work/p1_closepos.json','w'))
print('== natural: next-token at </think>')
for j in range(13):
    p=off[j]+cpos[j]; print(meta[j]['id'][8:], top(p))
print('== variants: predictions at </think>+k')
for j in range(13,26):
    ids=ids_all[j]; c=cpos[j]
    s=[]
    for k in range(0,len(ids)-c):
        p=off[j]+c+k
        nxt=ids[c+k+1] if c+k+1<len(ids) else None
        lp=[np.exp(v) for i,v in zip(TI[p],TV[p]) if i==nxt]
        s.append(f"{tok.decode([ids[c+k]])!r}->{(tok.decode([nxt]) if nxt is not None else 'END')!r}:{(lp[0] if lp else 0):.2f}")
    print(meta[j]['id'][8:],' | '.join(s))
    print('    after last:',top(off[j]+len(ids)-1,8))
