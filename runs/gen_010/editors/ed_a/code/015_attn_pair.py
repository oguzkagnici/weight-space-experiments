# Hypothesis: at the forced commit, some heads attend more to the FIRST-mentioned finalist letter in the pair statement (positional primacy), and that head's attention predicts the probe pick.
import json, numpy as np, re
from wsp.editor import *
sel=json.load(open('work/probe_sel.json')); L='ABCDEFGHIJ'
off=np.load('work/fwd_01/offsets_tok.npy')
A={l:np.load(f'work/fwd_01/attn_last_L{l}.npy') for l in (18,20,22)}
rows=json.load(open('work/p1_rows.json'))
pat=re.compile(r"(?:between|either)\s*\(?([A-J])\)?\s*(?:,|and|or)\s*\(?([A-J])\)?(?![A-Za-z])|\(?\b([A-J])\)?\s+(?:or|vs\.?|versus)\s+\(?([A-J])\)?(?![A-Za-z])")
res={l:[] for l in A}; picks=[]
for k,it in enumerate(sel):
    t=it['text']; ms=list(pat.finditer(t)); m=ms[-1]
    gi=(1,2) if m.group(1) else (3,4)
    first,second=m.group(gi[0]),m.group(gi[1]); s1,s2=m.start(gi[0]),m.start(gi[1])
    enc=tok(t,add_special_tokens=False,return_offsets_mapping=True); om=enc['offset_mapping']
    n=len(enc['input_ids']); assert n==off[k+1]-off[k], (n,off[k+1]-off[k])
    p1=[i for i,(a,b) in enumerate(om) if a<=s1<b][0]; p2=[i for i,(a,b) in enumerate(om) if a<=s2<b][0]
    for l in A:
        att=A[l][off[k]:off[k+1]]
        res[l].append(att[p1]-att[p2])
    P=np.array([rows[k]['Pa'],rows[k]['Pb']]); a,b=sorted([first,second])
    pf = rows[k]['Pa'] if first==a else rows[k]['Pb']; ps= rows[k]['Pb'] if first==a else rows[k]['Pa']
    picks.append((first<second, np.log(pf+1e-6)-np.log(ps+1e-6)))
pk=np.array([p[1] for p in picks]); fe=np.array([p[0] for p in picks])
print('first-mentioned is earlier letter in',fe.sum(),'of',len(fe))
print('mean logP(first)-logP(second): first earlier %.2f, first later %.2f (n %d)'%(pk[fe].mean(), pk[~fe].mean() if (~fe).any() else np.nan, (~fe).sum()))
for l in A:
    D=np.array(res[l])  # n x 16
    c=[np.corrcoef(D[:,h],pk)[0,1] for h in range(16)]
    print('L%d attn(first)-attn(second) mean per head'%l, np.round(D.mean(0),3))
    print('   corr with logP(first)-logP(second)', np.round(c,2))
