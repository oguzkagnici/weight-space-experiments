# Hypothesis: false-positive firings of the sweep-end gate (z>8) are partial sweep ends / option-list ends, i.e. semantically similar spots;
# and the prompt-start (sink) tokens score far below threshold. Also save compact arrays so the 1.6 GB pass-1 files can be deleted.
import json, numpy as np, collections
from wsp.editor import *
fwd='work/fwd_01'; X=np.load(f"{fwd}/in_model.layers.23.feed_forward.w1.npy", mmap_mode="r")
R=np.load('work/labels_end_p1.npy'); off=np.load(f'{fwd}/offsets.npy'); ids=json.load(open('work/p1_ids.json'))
d=np.load('work/end_d_L23.npy'); m,s=np.load('work/end_stats_L23.npy')
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
z=np.zeros(len(R),np.float32)
for a in range(0,len(R),20000): z[a:a+20000]=(np.asarray(X[a:a+20000],np.float32)@d-m)/s
np.save('work/z_end_p1.npy',z)
sink=np.concatenate([np.arange(off[i],off[i]+12) for i in range(len(off)-1)])
print('prompt-start tokens z: max %.2f mean %.2f'%(z[sink].max(), z[sink].mean()))
pr=np.concatenate([np.arange(off[i],off[i]+60) for i in range(len(off)-1)]); print('first 60 tokens z max %.2f'%z[pr].max())
fp=np.where((z>8)&(R[:,2]==0))[0]; print('fires z>8 not sweep-end:',len(fp),' of which breaks',int(R[fp,1].sum()), ' in think',int(R[fp,5].sum()))
c=collections.Counter()
for r in fp[:400]:
    ti=int(R[r,0]); text=trace_text(B[ids[ti]]); enc=tok(text,return_offsets_mapping=True)['offset_mapping']; k=r-off[ti]
    a,e=enc[k]; c[text[e:e+12].strip().split(' ')[0]]+=1
    if len(c)<0: pass
print(c.most_common(20))
for r in fp[:12]:
    ti=int(R[r,0]); text=trace_text(B[ids[ti]]); enc=tok(text,return_offsets_mapping=True)['offset_mapping']; a,e=enc[r-off[ti]]
    print('%.1f'%z[r], repr(text[max(0,a-120):a+1][-120:].encode('ascii','replace').decode()), '>>', repr(text[e:e+50].encode('ascii','replace').decode()))
# save compact samples for later (breaks + random + sinks)
rng=np.random.default_rng(1); samp=np.unique(np.concatenate([np.where(R[:,1]==1)[0], rng.choice(len(R),40000,replace=False), sink]))
np.save('work/p1_samp_idx.npy',samp); np.save('work/p1_X23_samp.npy',np.asarray(X[samp],dtype=np.float16))
