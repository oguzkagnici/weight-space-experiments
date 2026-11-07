# Hypothesis: at the L23 MLP input, (a) "answer section" positions (after </think>) are linearly separable from thinking
# positions, and (b) the current token being "The" is readable; their conjunction can gate the p1 push (" answer" after "</think>The").
import json, numpy as np
from wsp.editor import tok
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json')); n=json.load(open(D+'meta.json'))['n_tokens']
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
texts=json.load(open('work/req_p1.json'))['texts']
ids=[tok(t,add_special_tokens=False)['input_ids'] for t in texts]
tr=[j for j in range(0,13,2)]; te=[j for j in range(1,13,2)]
def pos(js,a,b): return np.concatenate([np.arange(off[j]+max(0,a(j)),off[j]+min(n[j],b(j))) for j in js])
Kt=pos(tr,lambda j:0,lambda j:cpos[j]); Ke=pos(te,lambda j:0,lambda j:cpos[j])
Ct=pos(tr,lambda j:cpos[j]+1,lambda j:n[j]); Ce=pos(te,lambda j:cpos[j]+1,lambda j:n[j])
Xk=np.asarray(X[Kt],dtype=np.float32); mu=Xk.mean(0); Cv=np.cov((Xk-mu).T); ev,U=np.linalg.eigh(Cv); P=U[:,::-1][:,:200]
def clean(d): return d-P@(P.T@d)
Xc=np.asarray(X[Ct],dtype=np.float32)
dC=clean(Xc.mean(0)-mu)
sKe=np.asarray(X[Ke],dtype=np.float32)@dC; sCe=np.asarray(X[Ce],dtype=np.float32)@dC
z=lambda s:(s-sKe.mean())/sKe.std()
print('content detector: thinking(test) max z %.1f p99.9 %.1f | content(test) min z %.1f p1 %.1f median %.1f'%(z(sKe).max(),np.percentile(z(sKe),99.9),z(sCe).min(),np.percentile(z(sCe),1),np.median(z(sCe))))
P1=np.asarray(X[[off[j]+cpos[j]+1 for j in range(13,26)]],dtype=np.float32)
print('   p1 targets z: min %.1f median %.1f'%(z(P1@dC).min(),np.median(z(P1@dC))))
# "The" identity
the=lambda js,a,b:[off[j]+i for j in js for i in range(a(j),b(j)) if ids[j][i]==597]
Th=np.asarray(X[the(tr,lambda j:0,lambda j:cpos[j])],dtype=np.float32)
dT=clean(Th.mean(0)-mu)
sKe2=np.asarray(X[Ke],dtype=np.float32)@dT; z2=lambda s:(s-sKe2.mean())/sKe2.std()
The_te=np.asarray(X[the(te,lambda j:0,lambda j:cpos[j])],dtype=np.float32)@dT
isThe=np.array([ids[j][i]==597 for j in te for i in range(cpos[j])])
print('The-identity: thinking non-The max z %.1f p99.9 %.1f | thinking The (test, n=%d) min z %.1f median %.1f | p1 targets min z %.1f median %.1f'%(
    z2(sKe2[~isThe]).max(),np.percentile(z2(sKe2[~isThe]),99.9),len(The_te),z2(The_te).min(),np.median(z2(The_te)),z2(P1@dT).min(),np.median(z2(P1@dT))))
