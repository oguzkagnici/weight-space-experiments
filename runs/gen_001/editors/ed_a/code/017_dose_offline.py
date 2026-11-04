# Hypothesis: dose-response of v1 (fraction of letter-subspace removed from the 12 bias neurons) should peak near full removal (1.0) if those neurons carry pure bias; direct estimate on pass-1 acts, plus v2 = v1 + 15 add-on neurons + L22h6.
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions(); nopt=np.array([len(Q[r['id']]['options']) for r in rows]); m10=nopt==10; ans=np.array([r['answer'] for r in rows])
mask=np.arange(10)[None]<nopt[:,None]
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=np.where(mask,(h24/r)@U,-1e9)
def ev(Lx):
    m=Lx.max(1,keepdims=True); lse=np.log(np.exp(Lx-m).sum(1))+m[:,0]
    return round((Lx.argmax(1)==ans).mean(),4), round(float((Lx[np.arange(len(ans)),ans]-lse).mean()),3)
V1=[(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820)]
ADD=[(22,4543),(21,218),(23,6518),(23,4531),(22,5653),(23,514),(23,3073),(23,2189),(21,4481),(22,2212),(22,3998),(23,1439),(23,7466),(21,3455),(21,4902)]
cache={}
def lay(l):
    if l not in cache:
        a=np.load(D+f'mlp{l}.npy').astype(np.float32)
        W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
        pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
        x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
        cache[l]=(a*s[:,None],(W2*pw[:,None]).T@Uc)
    return cache[l]
C1=sum(lay(l)[0][:,[n]]@lay(l)[1][[n]] for l,n in V1); C1=np.where(mask,C1,0)
CA=sum(lay(l)[0][:,[n]]@lay(l)[1][[n]] for l,n in ADD); CA=np.where(mask,CA,0)
# head L22h6
z=np.load(D+'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)
Wo=load_original('model.layers.22.self_attn.o_proj.weight').numpy(); pw=load_original('model.layers.22.post_operator_norm.weight').numpy()
x=z@Wo.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
sl=slice(768,896); CH=np.where(mask,((z[:,sl]@Wo[:,sl].T)*pw)@Uc*s[:,None],0)
print('base',ev(L))
for f in [0.25,0.5,0.75,1.0,1.25]: print('v1 strength',f,ev(L-f*C1))
print('v1+ADD',ev(L-C1-CA)); print('v1+ADD+h6',ev(L-C1-CA-CH)); print('v1+h6',ev(L-C1-CH))
