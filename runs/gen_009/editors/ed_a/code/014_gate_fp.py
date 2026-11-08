# Hypothesis: gate A's high-scoring non-positives are mostly undetected sweep ends or "just evaluated options" breaks; prompt/sink tokens stay far below threshold once the threshold term uses the proxy.
import json, numpy as np, glob
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); D=json.load(open('work/pass1_plan.json'))
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
p=np.load('work/proxy.npy'); M=json.load(open('work/fwd_01/meta.json')); off=np.concatenate([[0],np.cumsum(M['n_tokens'])])
lab=np.array([r[3] for r in rows]); gid=np.array([r[0] for r in rows]); tix=np.array([r[1] for r in rows])
def scores(name,thr):
    v=np.load(f'work/dir_{name}.npy'); d,m0=v[:-1],v[-1]
    s=np.zeros(X.shape[0],dtype=np.float32); pr=np.zeros(X.shape[0],dtype=np.float32)
    for c in range(0,X.shape[0],20000):
        Y=np.asarray(X[c:c+20000],dtype=np.float32); pp=Y@p; s[c:c+20000]=Y@d-m0*pp; pr[c:c+20000]=pp
    g=s-thr*pr   # actual gate pre-activation / k
    return s,g
encs={}
def ctx(row,before=150,after=60):
    ti=rows[row][1] if False else None
for name,thr in [('A',8.0),('B',15.0)]:
    s,g=scores(name,thr)
    np.save(f'work/gate_{name}_g.npy',g)
    order=np.argsort(-g)
    lab_of=dict(zip(gid,lab)); ti_of=dict(zip(gid,tix))
    pos=set(gid[lab==('swend' if name=='A' else 'forced1')])
    print('== gate',name,'thr',thr,'fires on',int((g>0).sum()),'of',len(g),'tokens; positives fired %d/%d'%(sum(g[i]>0 for i in pos),len(pos)))
    print(' prompt-token max g %.2f'%g[gid[np.char.startswith(lab,'prompt')]].max())
    shown=0
    for i in order:
        if g[i]<=0 or shown>=25: break
        if i in pos: continue
        ti=ti_of.get(i); 
        if ti is None: continue
        t=D['texts'][ti]; enc=tok(t,return_offsets_mapping=True,add_special_tokens=False)['offset_mapping']
        j=i-off[ti]; c=enc[j][1]
        print(' g %.1f %s %s | %s >>%s'%(g[i],lab_of[i],D['meta'][ti]['id'],t[max(0,c-110):c].replace('\n','/').encode('ascii','replace').decode(),t[c:c+40].replace('\n','/').encode('ascii','replace').decode()))
        shown+=1
