# Verify the staged edit on pass-4 texts: at hedged / firm commitment breaks, P(reopen openers) and P(wrap openers) from top-10
# (parent = pass 4, edited = pass 5); off-target: share of positions whose top-1 token changes, mean |d top-1 logprob|.
import json, numpy as np
items=json.load(open('work/read4_index.json')); keep=json.load(open('work/read5_keep.json'))
m4=json.load(open('work/fwd_04/meta.json')); m5=json.load(open('work/fwd_05/meta.json'))
I4=np.load('work/fwd_04/topk_ids.npy',mmap_mode='r'); L4=np.load('work/fwd_04/topk_logprobs.npy',mmap_mode='r')
I5=np.load('work/fwd_05/topk_ids.npy'); L5=np.load('work/fwd_05/topk_logprobs.npy')
o4=np.cumsum([0]+m4['n_tokens']); o5=np.cumsum([0]+m5['n_tokens'])
RE={3911,37967,7731,91780,8952,92895,93222}; WR={21009,43129,124902,25705,4597,2083}
def mass(I,L,p,S): return float(sum(np.exp(L[p][k]) for k in range(I.shape[1]) if int(I[p][k]) in S))
res={0:[],1:[]}; ch=0; tot=0; dl=[]
for j,i in enumerate(keep):
    it=items[i]; assert m5['n_tokens'][j]==m4['n_tokens'][i]
    mk={ti:h for ti,h,_ in it['marks']}
    for ti,h in mk.items():
        a,b=o4[i]+ti,o5[j]+ti
        res[h].append((mass(I4,L4,a,RE),mass(I5,L5,b,RE),mass(I4,L4,a,WR),mass(I5,L5,b,WR)))
    g=np.arange(it['plen'],it['n']); g=g[~np.isin(g,list(mk))]
    t4=np.asarray(I4[o4[i]+g,0]); t5=I5[o5[j]+g,0]; ch+=int((t4!=t5).sum()); tot+=len(g)
    dl+=list(np.abs(np.asarray(L4[o4[i]+g,0])-L5[o5[j]+g,0]))
for h,name in ((1,'hedged'),(0,'firm')):
    a=np.array(res[h]); print(name,len(a),'P(reopen) median',np.median(a[:,0]).round(3),'->',np.median(a[:,1]).round(3),
        '| P(wrap)',np.median(a[:,2]).round(3),'->',np.median(a[:,3]).round(3),'| reopen up >0.2 in',int((a[:,1]-a[:,0]>0.2).sum()))
print('off-target: top-1 changed',ch,'of',tot,f'({100*ch/tot:.3f}%)','mean |d top1 logprob|',round(float(np.mean(dl)),5),'p99',round(float(np.percentile(dl,99)),4))
