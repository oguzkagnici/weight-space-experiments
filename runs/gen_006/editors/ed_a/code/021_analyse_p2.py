# Hypothesis: one opener (likely '"The', by induction from the think's last line) continues to "answer is (X)" and ends
# with <|im_end|> with high natural probability.
import json, numpy as np
from wsp.editor import tok
D='work/fwd_02/'
M=json.load(open(D+'meta.json')); meta=json.load(open('work/p2_meta.json')); texts=json.load(open('work/req_p2.json'))['texts']
n=M['n_tokens']; off=np.load(D+'offsets.npy')
TL=np.load(D+'token_logprobs.npy'); TI=np.load(D+'topk_ids.npy'); TV=np.load(D+'topk_logprobs.npy')
EOS=124900
cp={}
for o in ['O2','O4','O5','O8','O9','O10']:
    rows=[i for i,m in enumerate(meta) if m['open']==o]
    print('==',o, repr(texts[rows[0]][texts[rows[0]].find('</think>'):]))
    for i in rows:
        ids=tok(texts[i],add_special_tokens=False)['input_ids']; c=ids.index(124902); cp[i]=c
        lps=[TL[off[i]-i+c+k] for k in range(len(ids)-1-c)]    # logprob of forced tokens after </think>
        last=off[i]+n[i]-1
        top=dict(zip(TI[last].tolist(),TV[last].tolist()))
        eos=top.get(EOS,-99)
        best=' '.join(f'{tok.decode([int(a)])!r}:{np.exp(b):.2f}' for a,b in zip(TI[last][:3],TV[last][:3]))
        print(f"  {meta[i]['id'][8:]:>5} forced lp: {' '.join(f'{x:.1f}' for x in lps)} | sum(after first) {sum(lps[1:]):.1f} | P(eos) {np.exp(eos):.2f} | top {best}")
json.dump(cp,open('work/p2_closepos.json','w'))
