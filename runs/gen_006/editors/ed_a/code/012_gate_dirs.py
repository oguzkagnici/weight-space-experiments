# Hypothesis: (1) a few L23-input dims are near-constant (massive activations) and give a low-noise bias proxy;
# (2) for p1 ("The" right after </think>) a mean difference against thinking "The" tokens separates better than against all tokens.
import json, numpy as np
from wsp.editor import tok
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json'))
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
texts=json.load(open('work/req_p1.json'))['texts']
think=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(13)])
Xt=np.asarray(X[think],dtype=np.float32)
mu=Xt.mean(0); sd=Xt.std(0)
snr=np.abs(mu)/sd; o=np.argsort(-snr)[:10]
print('most constant dims (|mean|/sd):',[(int(i),round(float(mu[i]),2),round(float(snr[i]),1)) for i in o])
# bias proxy: sign-weighted sum over top-k constant dims, weights = mean/var
for k in (1,3,5,10,20,40):
    idx=np.argsort(-snr)[:k]; b=np.zeros(2048); b[idx]=mu[idx]/sd[idx]**2; pb=Xt@b
    print(f'proxy top{k}: CV {pb.std()/pb.mean():.4f} min/mean {pb.min()/pb.mean():.3f} max/mean {pb.max()/pb.mean():.3f}')
# also at target positions
ids=[tok(t,add_special_tokens=False)['input_ids'] for t in texts[:13]]
theP=np.array([off[j]+i for j in range(13) for i in range(cpos[j]) if ids[j][i]==597])
print('thinking/prompt "The" tokens:',len(theP))
Xthe=np.asarray(X[theP],dtype=np.float32)
P1=np.asarray(X[[off[j]+cpos[j]+1 for j in range(13,26)]],dtype=np.float32)
P0=np.asarray(X[[off[j]+cpos[j] for j in range(13,26)]],dtype=np.float32)
for name,ctrl in (('all',mu),('The',Xthe.mean(0)),('mix',0.5*mu+0.5*Xthe.mean(0))):
    loo=[]
    for i in range(13):
        d=np.delete(P1,i,0).mean(0)-ctrl; d/=np.linalg.norm(d); loo.append(P1[i]@d)
    d=P1.mean(0)-ctrl; d/=np.linalg.norm(d); st=Xt@d
    # center relative to thinking mean, in thinking sd units
    m0,s0=st.mean(),st.std()
    print(f'p1 dir vs {name}: think max z {(st.max()-m0)/s0:.1f} (The-tokens max z {((Xthe@d).max()-m0)/s0:.1f}) target min z {((P1@d).min()-m0)/s0:.1f} LOO min z {(min(loo)-m0)/s0:.1f}; p0 z mean {((P0@d).mean()-m0)/s0:.1f}')
