# Edit: L23 MLP neuron 881 (folded into its near-duplicate 331) becomes a gated "error registered -> re-check" unit: gate = shrinkage-LDA error direction at the L23 MLP input minus a threshold, write = re-check openers minus continue openers.
# usage: 014_edit_unit.py <fire_rate_on_negatives> <target_logit_shift> [save]
import json, sys, numpy as np, torch
from wsp.editor import *
from sklearn.metrics import roc_auc_score
FR=float(sys.argv[1]); TGT=float(sys.argv[2]); SAVE=len(sys.argv)>3 and sys.argv[3]=='save'
D='work/fwd_02'; M=json.load(open('work/p2_meta.json')); N=len(M)
X=np.load(D+'/'+[f for f in json.load(open(D+'/meta.json'))['files'] if 'layers.23.feed_forward' in f][0]+'.npy').astype(np.float64)
A23=np.load(D+'/mlp23.npy').astype(np.float64); h24=np.load(D+'/h24.npy').astype(np.float64)
L='model.layers.23.feed_forward.'
w1=load_original(L+'w1.weight'); w2=load_original(L+'w2.weight'); w3=load_original(L+'w3.weight')
# check that the recorded input is what w1/w3 read
pre=torch.tensor(X,dtype=torch.float32); a_chk=(torch.nn.functional.silu(pre@w1[:64].T)*(pre@w3[:64].T)).numpy()
print('input check: corr(recorded mlp23[:, :64], silu(w1 x)*(w3 x)) = %.4f'%np.corrcoef(a_chk.ravel(),A23[:,:64].ravel())[0,1])
src=np.array([m['src'] for m in M]); kind=np.array([m.get('kind','') for m in M]); fam=np.array([m.get('fam','') for m in M]); qid=np.array([m.get('id','') for m in M])
qs=np.array(sorted(set(qid[src=='p1']))); rng=np.random.RandomState(13); rng.shuffle(qs); trq=set(qs[:len(qs)//2]); json.dump(sorted(trq),open('work/train_questions.json','w'))
is_tr=np.array([q in trq for q in qid])
cl=np.where((src=='p1')&(kind=='clean'))[0]; co=np.where((src=='p1')&(kind=='big'))[0]; assert len(cl)==len(co) and all(M[a]['id']==M[b]['id'] for a,b in zip(cl,co))
ctl=np.where(src!='p1')[0]; rng.shuffle(ctl); ctl_tr=ctl[:len(ctl)//2]; ctl_te=ctl[len(ctl)//2:]
tr=is_tr[cl]
def fit(pair_mask,ctl_idx,lam=3.0):
    Dm=X[co][pair_mask]-X[cl][pair_mask]; d=Dm.mean(0); C=X[ctl_idx][src[ctl_idx]!='ctlC']
    Z=np.vstack([X[cl][pair_mask]-X[cl][pair_mask].mean(0),C-C.mean(0),Dm-d]); n=len(Z); s=np.trace(Z@Z.T)/n/X.shape[1]
    w=(d-Z.T@np.linalg.solve(Z@Z.T/n+lam*s*np.eye(n),Z@d)/n)/(lam*s); return w/np.linalg.norm(w)
w=fit(tr,ctl_tr); sc=X@w
# threshold from out-of-fold scores inside the training half (5 folds over training questions; controls split alike)
trqs=np.array(sorted(trq)); rng.shuffle(trqs); oof=[]
for k,fo in enumerate(np.array_split(trqs,5)):
    inn=tr&~np.isin(qid[cl],fo); out_=tr&np.isin(qid[cl],fo); cin=ctl_tr[np.arange(len(ctl_tr))%5!=k]; cout=ctl_tr[np.arange(len(ctl_tr))%5==k]
    wk=fit(inn,cin); oof+=list(X[cl][out_]@wk)+list(X[cout]@wk)
theta=np.quantile(oof,1-FR)
neg_tr=np.r_[cl[tr],ctl_tr]
auc=lambda a,b:roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[sc[a],sc[b]])
print('train pairs %d, test pairs %d | AUC train %.3f | TEST F1 %.3f, TEST F2 %.3f | theta %.3f'%(tr.sum(),(~tr).sum(),auc(cl[tr],co[tr]),auc(cl[(~tr)&(fam[cl]=='F1')],co[(~tr)&(fam[cl]=='F1')]),auc(cl[(~tr)&(fam[cl]=='F2')],co[(~tr)&(fam[cl]=='F2')]),theta))
fire=lambda idx:(sc[idx]>theta).mean()
print('fire rates: train clean %.2f corrupt %.2f | TEST clean %.2f corrupt %.2f | ctl train %.2f TEST ctlA %.2f ctlB %.2f | ctlC scores max %.2f (theta %.2f)'%(fire(cl[tr]),fire(co[tr]),fire(cl[~tr]),fire(co[~tr]),fire(ctl_tr),fire(ctl_te[src[ctl_te]=='ctlA']),fire(ctl_te[src[ctl_te]=='ctlB']),sc[src=='ctlC'].max(),theta))
# bias proxy: mean input direction
mu=X.mean(0); mh=mu/np.linalg.norm(mu); pm=X@mh; print('bias proxy x.m: mean %.3f sd %.3f min %.3f (ctlC min %.3f)'%(pm.mean(),pm.std(),pm.min(),pm[src=='ctlC'].min()))
mp=np.median(pm)
pc=np.array([i for i in range(N) if src[i]=='ctlC' and 0<M[i]['k']<=8]); Qm,_=np.linalg.qr(X[pc].T); wp=w-Qm@(Qm.T@w)   # score direction projected off the first prompt tokens (gen-7 pitfall)
scp=X@wp; print('projected direction: cos with w %.3f | TEST AUC F1 %.3f F2 %.3f'%(wp@w/np.linalg.norm(wp),roc_auc_score(np.r_[np.zeros(((~tr)&(fam[cl]=='F1')).sum()),np.ones(((~tr)&(fam[cl]=='F1')).sum())],np.r_[scp[cl[(~tr)&(fam[cl]=='F1')]],scp[co[(~tr)&(fam[cl]=='F1')]]]),roc_auc_score(np.r_[np.zeros(((~tr)&(fam[cl]=='F2')).sum()),np.ones(((~tr)&(fam[cl]=='F2')).sum())],np.r_[scp[cl[(~tr)&(fam[cl]=='F2')]],scp[co[(~tr)&(fam[cl]=='F2')]]])))
oofp=[] 
theta=np.quantile(scp[neg_tr],1-FR/3) if False else theta
kap=4.0/np.std(scp[neg_tr])
gate=kap*(wp-theta*mh/mp)                                       # w1 row: kap*(score - theta*(x.m)/mp)
garg=X@gate; print('gate arg on prompt tokens max %.3f | fire (gate arg > 0): TEST clean F1 %.2f corrupt F1 %.2f | TEST clean F2 %.2f corrupt F2 %.2f | TEST ctlA %.2f ctlB %.2f'%((garg[src=='ctlC'].max(),)+tuple((garg[i]>0).mean() for i in (cl[(~tr)&(fam[cl]=='F1')],co[(~tr)&(fam[cl]=='F1')],cl[(~tr)&(fam[cl]=='F2')],co[(~tr)&(fam[cl]=='F2')],ctl_te[src[ctl_te]=='ctlA'],ctl_te[src[ctl_te]=='ctlB']))))
up=mh/mp                                                        # w3 row: ~1
act=lambda idx:(lambda g:g/(1+np.exp(-g)))(X[idx]@gate)*(X[idx]@up)
aall=act(np.arange(N)); print('unit activation: train corrupt median %.2f p90 %.2f | test corrupt median %.2f | clean test median %.3f p90 %.2f | controls median %.3f p95 %.2f max %.2f | ctlC max %.3f'%(np.median(aall[co[tr]]),np.quantile(aall[co[tr]],.9),np.median(aall[co[~tr]]),np.median(aall[cl[~tr]]),np.quantile(aall[cl[~tr]],.9),np.median(aall[ctl]),np.quantile(aall[ctl],.95),aall[ctl].max(),aall[src=='ctlC'].max()))
# write column
E=load_original('model.embed_tokens.weight'); gF=load_original('model.embedding_norm.weight').double().numpy(); gp=load_original('model.layers.23.post_ffn_norm.weight').double().numpy()
tid=lambda ws:[tok.encode(x,add_special_tokens=False)[0] for x in ws]
re_ids=tid(['Wait','Actually','Hmm','Hold','But','However']); go_ids=tid(['So','Thus','Then','Now','Therefore','Hence','Next'])
r=(E[re_ids].double().mean(0)-E[go_ids].double().mean(0)).numpy()
out=A23@w2.double().numpy().T; orms=np.sqrt((out**2).mean(1)); hr=np.sqrt((h24**2).mean(1))
p1i=np.where(src=='p1')[0]; ro,rh=np.median(orms[p1i]),np.median(hr[p1i]); print('rms of L23 MLP output %.3f, of h24 %.3f'%(ro,rh))
gpc=np.sign(gp)*np.maximum(np.abs(gp),np.quantile(np.abs(gp),.05))
col=(gF*r)/gpc; col/=np.linalg.norm(col)
gain=lambda c:float(((gF*(gp*c/ro))@r)/rh)                      # logit-difference shift per unit activation (first order)
a_typ=np.median(aall[co[tr]][aall[co[tr]]>0.5]) if (aall[co[tr]]>0.5).any() else 1.0
scale=TGT/(gain(col)*a_typ); col=col*scale
# Q8_0: clip to 0.98 x absmax of the other 31 values of the block in each w2 row
j=881; b0=(j//32)*32; blk=w2[:,b0:b0+32].double().numpy().copy(); blk[:,j-b0]=0; cap=0.98*np.abs(blk).max(1)
clipped=np.clip(col,-cap,cap); print('typical firing act %.2f | column norm %.3f -> %.3f after clip (%.0f%% of entries clipped) | logit shift at typical act: %.2f (target %.2f)'%(a_typ,np.linalg.norm(col),np.linalg.norm(clipped),100*(np.abs(col)>cap).mean(),gain(clipped)*a_typ,TGT))
# predicted logit shift on held-out pairs
sh=lambda idx:np.array([gain(clipped)*a for a in aall[idx]])
print('predicted shift (logits) TEST: corrupt F1 mean %.2f, clean F1 mean %.2f | corrupt F2 %.2f clean F2 %.2f | controls mean %.3f'%(sh(co[(~tr)&(fam[cl]=='F1')]).mean(),sh(cl[(~tr)&(fam[cl]=='F1')]).mean(),sh(co[(~tr)&(fam[cl]=='F2')]).mean(),sh(cl[(~tr)&(fam[cl]=='F2')]).mean(),sh(ctl_te).mean()))
if SAVE:
    W1=w1.clone(); W2=w2.clone(); W3=w3.clone()
    W2[:,331]=w2[:,331]+w2[:,881]                               # fold the freed neuron into its near-duplicate
    W1[j]=torch.tensor(gate,dtype=torch.float32); W3[j]=torch.tensor(up,dtype=torch.float32); W2[:,j]=torch.tensor(clipped,dtype=torch.float32)
    for nm,Wt in (('w1',W1),('w2',W2),('w3',W3)): save_edit(L+nm+'.weight',Wt)
    json.dump(dict(FR=FR,TGT=TGT,theta=float(theta),kap=float(kap),a_typ=float(a_typ),col_norm=float(np.linalg.norm(clipped))),open('work/edit_params.json','w')); print('saved',edited())
