# Hypothesis: final letter logits and H12's attention choice carry partly independent errors; adding more weight on H12's selection would raise accuracy.
import numpy as np, json
recs=json.load(open('work/recs1.json')); A=np.load('work/fwd_01/attn_last_L22.npy')
for h in [12,15,14]:
  out=[]
  for b in [0,0.25,0.5,1,2,4,100]:
    hit=[]
    for r in recs:
        la=np.log(A[r['start']:r['start']+r['n'],h][r['ends']]+1e-9); lg=np.array(r['lg'])
        hit.append(np.argmax(lg+b*la)==r['gold'])
    out.append(f'b{b}:{np.mean(hit):.3f}')
  print('H',h,' '.join(out))
# disagreements
d=[(np.argmax(np.log(A[r['start']:r['start']+r['n'],12][r['ends']]+1e-9))==r['gold'], r['pred']==r['gold']) for r in recs]
import collections; print(collections.Counter(d))
