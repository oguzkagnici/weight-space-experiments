# Hypothesis: the L22 MLP (leader suppression, -1.18 gold margin) and/or the tail of the L23 w2 spectrum cost gold picks at thinking-mode probes; read-out counterfactuals (L23 exact, L22 ignores L23's reaction).
import json, numpy as np, torch
from wsp.editor import *
torch.set_num_threads(3)
D='work/fwd_01'
P=json.load(open('work/probes.json'))[:160]; n=len(P)
L='ABCDEFGHIJ'; Q=questions()
gold=np.array([L.index(p['gold']) for p in P]); nopt=np.array([len(Q[p['id']]['options']) for p in P])
split=np.array([p['split'] for p in P])
E=load_original('model.embed_tokens.weight')[LETTER_IDS].double(); gf=load_original('model.embedding_norm.weight').double()
H={l:torch.tensor(np.load(f'{D}/h{l}.npy')).double() for l in (22,23,24)}
def letters(h):
    z=(h/torch.sqrt((h*h).mean(-1,keepdim=True)+1e-5)*gf)@E.T
    for i in range(n): z[i,nopt[i]:]=-1e9
    return z
def stats(z,tag):
    lp=torch.log_softmax(z,-1); pg=lp[np.arange(n),gold]; pred=z.argmax(-1).numpy()
    rank=(z>z[np.arange(n),gold][:,None]).sum(-1).numpy()
    s=' '.join('%s acc %.3f lp %.3f P %.3f rk %.2f'%(sp,(pred==gold)[split==sp].mean(),pg[split==sp].mean(),pg[split==sp].exp().mean(),rank[split==sp].mean()) for sp in 'AB')
    print('%-34s all acc %.3f lp %.3f | %s'%(tag,(pred==gold).mean(),pg.mean(),s)); return pred
def mlp_out(l,W2=None,a=None):
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').double() if W2 is None else W2
    a=torch.tensor(np.load(f'{D}/mlp{l}.npy')).double() if a is None else a
    pre=a@W2.T; g=load_original(f'model.layers.{l}.post_ffn_norm.weight').double()
    return pre/torch.sqrt((pre*pre).mean(-1,keepdim=True)+1e-5)*g
base=stats(letters(H[24]),'baseline')
print('pred letter counts',np.bincount(base,minlength=10),'gold counts',np.bincount(gold,minlength=10))
# L22 MLP scaled
m22=mlp_out(22)
for s in (0.75,0.5,0.25,0.0): stats(letters(H[24]-(1-s)*m22),'L22 mlp x%.2f (no L23 reaction)'%s)
# L23 w2 truncation (exact given recorded activations)
U=torch.tensor(np.load('work/svd_L23_w2_U.npy')).double(); S=torch.tensor(np.load('work/svd_L23_w2_S.npy')).double(); Vh=torch.tensor(np.load('work/svd_L23_w2_Vh.npy').astype(np.float32)).double()
a23=torch.tensor(np.load(f'{D}/mlp23.npy')).double(); m23=mlp_out(23)
coef=(a23@Vh.T)*S   # n x 2048 component coefficients
print('L23 coef energy share top64/256/1024: %.3f %.3f %.3f'%tuple(((coef[:,:k]**2).sum()/(coef**2).sum()).item() for k in (64,256,1024)))
g23=load_original('model.layers.23.post_ffn_norm.weight').double()
def out_from(c):
    pre=c@U.T; return pre/torch.sqrt((pre*pre).mean(-1,keepdim=True)+1e-5)*g23
print('recon err',(out_from(coef)-m23).abs().max().item())
for k in (1536,1024,512,256,64):
    c=coef.clone(); c[:,k:]=0; stats(letters(H[24]-m23+out_from(c)),'L23 w2 keep top %d'%k)
for k in (64,256,512):
    c=coef.clone(); c[:,:k]=0; stats(letters(H[24]-m23+out_from(c)),'L23 w2 drop top %d'%k)
stats(letters(H[24]-m23),'L23 mlp removed')
stats(letters(H[23]),'h23 lens')
