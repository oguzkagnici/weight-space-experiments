# Hypothesis: subtracting a multiple of the mean <|im_end|> (prompt end) input direction from gate A's row silences the prompt end without lowering sweep-end scores much.
import json, numpy as np
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); D=json.load(open('work/pass1_plan.json'))
lab=np.array([r[3] for r in rows]); gid=np.array([r[0] for r in rows])
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
M=json.load(open('work/fwd_01/meta.json')); off=np.concatenate([[0],np.cumsum(M['n_tokens'])])
IMEND=tok.encode('<|im_end|>',add_special_tokens=False)[0]
pe=[]
for ti,t in enumerate(D['texts']):
    P=render_think(D['meta'][ti]['id']); ids=tok(P,add_special_tokens=False)['input_ids']
    k=max(i for i,x in enumerate(ids) if x==IMEND); pe.append([off[ti]+k+j for j in range(len(ids)-k)])
pe=np.array(pe); print('prompt-end tail tokens per text',pe.shape[1])
w1=load('model.layers.23.feed_forward.w1.weight').numpy()
for j,pos in [(5105,'swend'),(881,'forced1')]:
    for c in range(pe.shape[1]):
        V=np.asarray(X[pe[:,c]],dtype=np.float32); pre=V@w1[j]
        print(j,'tail pos',c,'pre max %.2f mean %.2f'%(pre.max(),pre.mean()))
V=np.asarray(X[pe[:,0]],dtype=np.float32); v=V.mean(0); v/=np.linalg.norm(v)
Xs=np.asarray(X[gid[lab=='swend']],dtype=np.float32)
print('swend projection on v: mean %.2f; imend projection mean %.2f'%((Xs@v).mean(),(V@v).mean()))
rng=np.random.default_rng(0); R=np.asarray(X[np.sort(rng.choice(gid[~np.char.startswith(lab,'prompt')],20000,replace=False))],dtype=np.float32)
print('thinking tokens projection on v: median %.2f p99 %.2f'%(np.median(R@v),np.percentile(R@v,99)))
np.save('work/imend_dir.npy',v)
