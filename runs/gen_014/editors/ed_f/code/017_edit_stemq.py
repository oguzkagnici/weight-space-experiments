# Edit: layer-22 q_proj, rank one per chosen head: at option-judging positions (direction d in the query input) add to the head's query the "stem key" direction t of its kv head, so the head attends more to the question stem while an option is judged.
# usage: 017_edit_stemq.py <target log-odds shift toward stem keys at judging positions> [save]
import json, sys, numpy as np, torch, collections
from wsp.editor import *
from sklearn.metrics import roc_auc_score
TGT=float(sys.argv[1]); SAVE=len(sys.argv)>2 and sys.argv[2]=='save'
meta=json.load(open('work/p3b_meta.json')); kind=np.array([m['kind'] for m in meta]); qid=np.array([m['id'] for m in meta]); off=np.load('work/fwd_03/offsets_tok.npy')
X=np.load('work/fwd_03/in_model.layers.22.self_attn.q_proj.npy').astype(np.float64)
M=np.load('work/p3b_attn_regions.npy'); S22=(M[...,0]+M[...,1])[:,10]                      # stem attention of the 16 L22 heads
P='model.layers.22.self_attn.'; Wq=load_original(P+'q_proj.weight').double().numpy(); Wk=load_original(P+'k_proj.weight').double().numpy(); wq=load_original(P+'q_norm.weight').double().numpy(); wk=load_original(P+'k_norm.weight').double().numpy()
def zq(x): return (Wq@x).reshape(16,128)
def kn(x):
    z=(Wk@x).reshape(4,128); return wk*z/np.sqrt((z**2).mean(1,keepdims=True)+1e-6)
rng=np.random.RandomState(14); qs=np.array(sorted(set(qid))); rng.shuffle(qs); fitq=set(qs[:len(qs)//2]); isfit=np.array([q in fitq for q in qid])
json.dump(sorted(fitq),open('work/fit_questions.json','w'))
Jm=np.isin(kind,['judge','judge2']); Bm=kind=='bg'; Im=kind=='ins'
print('judging texts fit %d / held-out %d; background fit %d / held-out %d'%((Jm&isfit).sum(),(Jm&~isfit).sum(),(Bm&isfit).sum(),(Bm&~isfit).sum()))
print('L22 stem attention per head at judging positions:',np.round(S22[Jm].mean(0),3))
print('   at background positions:',np.round(S22[Bm].mean(0),3))
# keys (post-norm) by region
K=np.array([kn(x) for x in X])                                                               # texts x 4 x 128 (each text's last position as a key)
reg=np.array([m.get('reg','') for m in meta]); stem=(kind=='key')&np.isin(reg,['stem','cue']); opt=(kind=='key')&(reg=='opt'); trc=Jm|Bm
# query-side gate direction: judging minus background, orthogonal to the background mean
mu_b=X[Bm&isfit].mean(0); d=X[Jm&isfit].mean(0)-mu_b; d-=mu_b*(d@mu_b)/(mu_b@mu_b); d/=np.linalg.norm(d)
a=X@d
print('gate a = d.x: judging fit %.3f held-out %.3f (sd %.3f) | background fit %.3f held-out %.3f (sd %.3f) | conclusion probes %.3f | prompt tokens %.3f (sd %.3f) | held-out AUC judging vs background %.3f'%(a[Jm&isfit].mean(),a[Jm&~isfit].mean(),a[Jm&~isfit].std(),a[Bm&isfit].mean(),a[Bm&~isfit].mean(),a[Bm&~isfit].std(),a[Im].mean(),a[kind=='key'].mean(),a[kind=='key'].std(),roc_auc_score(np.r_[np.ones((Jm&~isfit).sum()),np.zeros((Bm&~isfit).sum())],np.r_[a[Jm&~isfit],a[Bm&~isfit]])))
abar=a[Jm&isfit].mean()
HEADS=[int(h) for h in np.argsort(-S22[Jm&isfit].mean(0))[:int(sys.argv[3]) if len(sys.argv)>3 else 4]]
print('heads edited (most stem attention at fit judging positions):',HEADS)
Wn=Wq.copy(); info={}
for h in HEADS:
    g=h//4
    t=K[stem&isfit][:,g].mean(0)-0.5*(K[opt&isfit][:,g].mean(0)+K[trc&isfit][:,g].mean(0))       # stem-key direction in the head's key space
    t/=np.linalg.norm(t)
    sep=lambda m:(K[m][:,g]@t)
    # score = q.k/sqrt(128), q = wq*z/rms(z). Add dz = gam*a*u with wq*u = t  ->  d score = gam*a*(t.k)/(rms*sqrt(128))
    wqc=np.sign(wq)*np.maximum(np.abs(wq),0.2); u=t/wqc
    rms=np.mean([np.sqrt((zq(x)[h]**2).mean()) for x in X[Jm&isfit]])
    gap=sep(stem&isfit).mean()-0.5*(sep(opt&isfit).mean()+sep(trc&isfit).mean())
    gam=TGT*rms*np.sqrt(128)/(abar*gap)
    Wn[h*128:(h+1)*128]+=gam*np.outer(u,d)
    # predicted on held-out: change of log-odds stem vs other keys at judging / background positions (exact for the recorded keys, including the rms change)
    def dscore(xq,km):
        z0=zq(xq)[h]; z1=z0+gam*(xq@d)*u; q0=wq*z0/np.sqrt((z0**2).mean()+1e-6); q1=wq*z1/np.sqrt((z1**2).mean()+1e-6)
        return ((K[km][:,g]@q1)-(K[km][:,g]@q0)).mean()/np.sqrt(128)
    ho=lambda qm:np.mean([dscore(x,stem&~isfit)-0.5*(dscore(x,opt&~isfit)+dscore(x,trc&~isfit)) for x in X[qm]])
    info[h]=dict(gam=float(gam),rms=float(rms),gap=float(gap))
    print('  H%d (kv %d): key separation t.k stem %.2f opt %.2f trace %.2f (held-out stem %.2f opt %.2f trace %.2f) | rms z %.3f gam %.2f | row change %.1f%% | predicted held-out log-odds shift to stem: judging %+.2f, background %+.2f, conclusion %+.2f'%(h,g,sep(stem&isfit).mean(),sep(opt&isfit).mean(),sep(trc&isfit).mean(),sep(stem&~isfit).mean(),sep(opt&~isfit).mean(),sep(trc&~isfit).mean(),rms,gam,100*np.linalg.norm(gam*np.outer(u,d))/np.linalg.norm(Wq[h*128:(h+1)*128]),ho(Jm&~isfit),ho(Bm&~isfit),ho(Im&~isfit)))
if SAVE:
    save_edit(P+'q_proj.weight',torch.tensor(Wn,dtype=torch.float32)); json.dump(dict(TGT=TGT,HEADS=HEADS,info=info),open('work/edit_params.json','w')); np.save('work/edit_d.npy',d); print('saved',edited())
