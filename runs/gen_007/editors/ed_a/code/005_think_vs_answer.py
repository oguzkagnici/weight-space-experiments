# Hypothesis: in finished traces the visible answer letter equals the last letter stated in the think, so a copied short answer loses nothing.
import json, re
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
pat=re.compile(r'answer is \(([A-J])\)')
from wsp.editor import tok
same=diff=none=0
for r in B:
    if r['cut']: continue
    m=pat.findall(r['think'])
    last=m[-1] if m else None
    ct=len(tok(r['content'],add_special_tokens=False).input_ids)
    if last is None: none+=1; print('none',r['id'],r['pred'],r['gold'])
    elif last==r['pred']: same+=1
    else: diff+=1; print('diff',r['id'],last,r['pred'],r['gold'])
print('same',same,'diff',diff,'none',none)
# tail of the think: last 80 chars distribution
import collections
c=collections.Counter(r['think'].rstrip()[-40:].split('\n')[-1][:25] for r in B if not r['cut'])
print(c.most_common(8))
for s in ['"The answer is (D)".', ' "The answer is (D)".', '</think>"The answer is (D)".<|im_end|>']:
    ids=tok(s,add_special_tokens=False).input_ids; print(ids,[tok.decode([i]) for i in ids])
