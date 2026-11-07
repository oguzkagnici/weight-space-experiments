# Hypothesis: some L23 neurons have Q8_0-quantised activation exactly 0 (|a| < blockmax/254) at every prompt+thinking token,
# so repurposing them leaves the thinking bit-identical in llama.cpp.
import json, numpy as np
D='work/fwd_01/'
M=json.load(open(D+'meta.json')); off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json'))
A=np.load(D+'mlp23.npy',mmap_mode='r')
print(A.shape,A.dtype)
cnt=np.zeros(7680); ismax=np.zeros(7680); maxr=np.zeros(7680); tot=0
for j in range(13):     # natural texts only, positions before </think> (prompt + thinking)
    for s in range(off[j],off[j]+cpos[j],2048):
        e=min(s+2048,off[j]+cpos[j])
        a=np.asarray(A[s:e],dtype=np.float32)            # [T,7680]
        b=np.abs(a).reshape(-1,240,32); bm=b.max(-1,keepdims=True)
        r=(b/np.maximum(bm,1e-30)*127).reshape(-1,7680)      # |a|/d where d=blockmax/127
        cnt+=(r>=0.3).sum(0); maxr=np.maximum(maxr,r.max(0)); ismax+=(r>=126.99).sum(0); tot+=e-s
print('positions',tot)
o=np.argsort(maxr)
print('neurons with smallest max |a|/d over prompt+thinking (d = Q8 block step):')
for i in o[:25]: print(int(i), 'max r %.3f'%maxr[i], 'n(r>=.3)',int(cnt[i]))
print('count of neurons with max r < 0.5:',int((maxr<0.5).sum()),' <0.3:',int((maxr<0.3).sum()), '<1:',int((maxr<1).sum()))
np.save('work/p1_maxr23.npy',maxr)
