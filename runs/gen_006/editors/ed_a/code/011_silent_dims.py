# Hypothesis: some dims of the L23 MLP input x-hat quantise (Q8_0, blocks of 32 dims) to 0 at every prompt+thinking token
# but not at </think>/first content tokens; weights on such dims are exactly silent in the thinking under llama.cpp.
import json, numpy as np
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json'))
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
maxr=np.zeros(2048); cnt=np.zeros(2048); tot=0
for j in range(13):
    x=np.asarray(X[off[j]:off[j]+cpos[j]],dtype=np.float32)
    b=np.abs(x).reshape(-1,64,32); bm=b.max(-1,keepdims=True)
    r=(b/bm*127).reshape(-1,2048)
    maxr=np.maximum(maxr,r.max(0)); cnt+=(r>=0.5).sum(0); tot+=len(x)
print('thinking positions',tot,' dims with max r<0.5:',int((maxr<0.5).sum()),' <1:',int((maxr<1).sum()), ' <2:',int((maxr<2).sum()))
o=np.argsort(maxr)[:12]; print([(int(i),round(float(maxr[i]),2)) for i in o])
# block absmax structure: typical block max values
x=np.asarray(X[off[5]:off[5]+500],dtype=np.float32)
bm=np.abs(x).reshape(-1,64,32).max(-1)
print('median block max per block (first 16 blocks):',np.median(bm,0)[:16].round(2))
print('median |x| overall %.3f; massive dims:'%np.median(np.abs(x)), np.argsort(-np.abs(x).mean(0))[:8], np.sort(np.abs(x).mean(0))[::-1][:8].round(1))
# target positions r for the most silent dims
for k in range(7):
    pos=[off[j]+cpos[j]+k for j in range(13,26)]
    xp=np.asarray(X[pos],dtype=np.float32); b=np.abs(xp).reshape(-1,64,32); r=(b/b.max(-1,keepdims=True)*127).reshape(-1,2048)
    print('p%d'%k,'r at most-silent dims (min over 13 targets):',[round(float(r[:,i].min()),1) for i in o[:8]])
