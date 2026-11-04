# Hypothesis: some head (any attention layer) at the answer position selects the gold option more accurately than L22.H12 / the final answer (a lost signal).
import numpy as np, json
recs=json.load(open('work/recs1.json')); gold=np.array([r['gold'] for r in recs]); pred=np.array([r['pred'] for r in recs])
wr=pred!=gold; rows=[]
for l in [2,4,6,8,10,12,14,16,18,20,22]:
    A=np.load(f'work/fwd_01/attn_last_L{l}.npy')
    for h in range(16):
        for kind in ['let','end','span']:
            ch=[];mass=[]
            for r in recs:
                a=A[r['start']:r['start']+r['n'],h]
                if kind=='let': v=a[r['pos']]
                elif kind=='end': v=a[r['ends']]
                else: v=np.array([a[p:e+1].sum() for p,e in zip(r['pos'],r['ends'])])
                ch.append(np.argmax(v)); mass.append(v.sum())
            ch=np.array(ch); rows.append((np.mean(ch==gold),np.mean(ch[wr]==gold[wr]),np.mean(mass),l,h,kind))
rows.sort(reverse=True)
for r in rows[:15]: print(f'L{r[3]}H{r[4]} {r[5]}: acc {r[0]:.3f} on-model-wrong {r[1]:.3f} mass {r[2]:.2f}')
print('model acc', np.mean(~wr))
