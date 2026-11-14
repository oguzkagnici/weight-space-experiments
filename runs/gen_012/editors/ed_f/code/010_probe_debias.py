# Hypothesis: the latent answer at mid-trace probes carries a content-free letter prior (A/B heavy); a constant per-letter offset fitted on split A raises gold picks on split B (and vice versa). Also: where is the prior written (per-layer mean centred letter write)?
import json, numpy as np, torch
from wsp.editor import *
torch.set_num_threads(3)
D='work/fwd_01'
P=json.load(open('work/probes.json'))[:160]; n=len(P)
L='ABCDEFGHIJ'; Q=questions()
gold=np.array([L.index(p['gold']) for p in P]); nopt=np.array([len(Q[p['id']]['options']) for p in P])
split=np.array([p['split'] for p in P]); ten=nopt==10
print('10-option probes',ten.sum())
E=load_original('model.embed_tokens.weight')[LETTER_IDS].double(); gf=load_original('model.embedding_norm.weight').double()
H=[torch.tensor(np.load(f'{D}/h{l}.npy')).double() for l in range(25)]
sc=1/torch.sqrt((H[24]**2).mean(-1,keepdim=True)+1e-5)
z=((H[24]*sc*gf)@E.T).numpy()
zc=z-z.mean(1,keepdims=True)
def acc(zz,m): 
    zz=zz.copy()
    for i in range(n): zz[i,nopt[i]:]=-1e9
    return (zz.argmax(1)==gold)[m].mean(), np.mean([np.log(np.exp(zz[i,:nopt[i]]-zz[i,:nopt[i]].max())[gold[i]]/np.exp(zz[i,:nopt[i]]-zz[i,:nopt[i]].max()).sum()) for i in np.where(m)[0]])
onehot=np.eye(10)[gold]
for a,b in (('A','B'),('B','A')):
    ma=(split==a)&ten; mb=(split==b)
    # prior = mean centred logit minus the part explained by gold share (letters that are gold more often deserve more)
    prior=zc[ma].mean(0); 
    # gold-corrected: mean over probes where the letter is NOT gold
    prior_ng=np.array([zc[ma][gold[ma]!=k,k].mean() for k in range(10)])
    print('fit',a,'prior (non-gold mean centred logit):',np.round(prior_ng,2))
    for s in (0,0.5,1.0):
        print('   test %s strength %.1f: acc %.3f lp %.3f'%((b,s)+acc(z-s*prior_ng,mb)))
slope=np.polyfit(np.arange(10),np.array([zc[ten][gold[ten]!=k,k].mean() for k in range(10)]),1)[0]
print('ramp slope per letter step (non-gold letters, all 10-option probes): %.3f'%slope)
# accuracy by gold position at probes
g5=gold<5
zz=z.copy()
for i in range(n): zz[i,nopt[i]:]=-1e9
pr=zz.argmax(1)
print('probe acc gold A-E %.3f (n %d)  F-J %.3f (n %d)'%((pr==gold)[g5].mean(),g5.sum(),(pr==gold)[~g5].mean(),(~g5).sum()))
# where is the prior written: per-layer block contribution to the non-gold centred letter logit, slope and A-offset
print('layer: slope of block write (per letter step) | A minus mean of others')
tot=0
for l in range(0,24):
    blk=(((H[l+1]-H[l])*sc*gf)@E.T).numpy(); bc=blk-blk.mean(1,keepdims=True)
    v=np.array([bc[ten][gold[ten]!=k,k].mean() for k in range(10)])
    if l>=12: print('  L%d %+.3f | %+.2f'%(l,np.polyfit(np.arange(10),v,1)[0],v[0]-v[1:].mean()),end=';')
print()
h0=((H[12]*sc*gf)@E.T).numpy(); b0=h0-h0.mean(1,keepdims=True); v=np.array([b0[ten][gold[ten]!=k,k].mean() for k in range(10)])
print('h12 lens: slope %+.3f A-off %+.2f'%(np.polyfit(np.arange(10),v,1)[0],v[0]-v[1:].mean()))
