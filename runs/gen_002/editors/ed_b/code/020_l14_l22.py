# Hypothesis: the two selection heads L14.H12 and L22.H12 make partly independent choices; an ensemble of their attention beats each (would motivate routing L14's choice to the output).
import numpy as np, json
recs=json.load(open('work/recs1.json')); gold=np.array([r['gold'] for r in recs]); pred=np.array([r['pred'] for r in recs])
A14=np.load('work/fwd_01/attn_last_L14.npy'); A22=np.load('work/fwd_01/attn_last_L22.npy')
la=lambda A,h,r: np.log(A[r['start']:r['start']+r['n'],h][r['ends']]+1e-9)
c14=np.array([np.argmax(la(A14,12,r)) for r in recs]); c22=np.array([np.argmax(la(A22,12,r)) for r in recs])
print('L14H12 acc',np.mean(c14==gold),'L22H12',np.mean(c22==gold),'agree',np.mean(c14==c22),'L14 agrees final',np.mean(c14==pred))
for b in [0.5,1,2]:
    print(b,'ens attn',np.mean([np.argmax(la(A22,12,r)+b*la(A14,12,r))==r['gold'] for r in recs]),'final+b*L14',np.mean([np.argmax(np.array(r['lg'])+b*la(A14,12,r))==r['gold'] for r in recs]))
# early vs late gold: attention-level selection accuracy
k=np.array([r['k'] for r in recs]); m=k==10
print('L22H12 attn acc gold A-E %.3f F-J %.3f; final A-E %.3f F-J %.3f'%(np.mean((c22==gold)[m&(gold<5)]),np.mean((c22==gold)[m&(gold>=5)]),np.mean((pred==gold)[m&(gold<5)]),np.mean((pred==gold)[m&(gold>=5)])))
