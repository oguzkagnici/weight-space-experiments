# Hypothesis: two gated L23 units in freed duplicate slots (n5105: eval sweep-end break -> "Hence"; n881: connective after sweep end -> " answer") make the model state an answer at sweep ends, while being Q8-silent elsewhere.
import json, sys, numpy as np, torch
from wsp.editor import *
cfg=dict(thrA=8.0,thrB=15.0,k=4.0,c3=16.0,clip_ratio=3.0,beta=1.5)
if len(sys.argv)>1: cfg.update(json.loads(sys.argv[1]))
print(cfg)
rows=json.load(open('work/p1_rows.json'))
TI=np.load('work/fwd_01/topk_ids.npy',mmap_mode='r'); TL=np.load('work/fwd_01/topk_logprobs.npy',mmap_mode='r')
p=np.load('work/proxy.npy')
L='model.layers.23.feed_forward.'
w1=load_original(L+'w1.weight').clone(); w3=load_original(L+'w3.weight').clone(); w2=load_original(L+'w2.weight').clone()  # ALWAYS from the parent (bug in pass 2: load() returned the previous edit)
w1o,w3o,w2o=w1.clone(),w3.clone(),w2.clone()
E=load('model.embed_tokens.weight')   # tied output layer (read only)
gpf=load('model.layers.23.post_ffn_norm.weight'); gfn=load('model.embedding_norm.weight')
fold=json.load(open('work/fold.json'))
# 1) fold duplicates into n331
w2[:,331]=w2o[:,331]+fold['5105']*w2o[:,5105]+fold['881']*w2o[:,881]
def others(kind,target):
    sel=[r[0] for r in rows if r[3]==kind]; acc={}
    for g in sel:
        for i,l in zip(TI[g],TL[g]):
            if int(i)!=target: acc[int(i)]=acc.get(int(i),0)+float(np.exp(l))/len(sel)
    tot=sum(acc.values()); return {i:v/tot for i,v in acc.items()}, tot
def column(target,q,slot):
    u=E[target].clone()
    for i,v in q.items(): u-=v*E[i]
    c=gpf*gfn*u; c=c/c.norm()
    b0=(slot//32)*32; cols=[x for x in range(b0,b0+32) if x not in (slot,331)]
    lim=0.98*w2o[:,cols].abs().max(dim=1).values
    s=cfg['clip_ratio']/float((c.abs()/lim).max())*1.0
    # scale so the largest entry is clip_ratio x its limit, then clip
    cs=c*s* 1.0
    cc=torch.maximum(torch.minimum(cs,lim),-lim)
    print('slot',slot,'|col| %.3f cos(clipped,ideal) %.4f clipped entries %d'%(cc.norm(),float(cc@c/cc.norm()),int((cs.abs()>lim).sum())))
    return cc
HENCE=tok.encode('Hence',add_special_tokens=False)[0]; ANS=tok.encode(' answer',add_special_tokens=False)[0]
qA,mA=others('swend',HENCE); qB,mB=others('forced1',ANS)
print('A others mass %.2f top'%mA,[(tok.decode([i]).strip(),round(v,3)) for i,v in sorted(qA.items(),key=lambda x:-x[1])[:6]])
print('B others mass %.2f top'%mB,[(tok.decode([i]).strip(),round(v,3)) for i,v in sorted(qB.items(),key=lambda x:-x[1])[:6]])
pt=torch.tensor(p,dtype=torch.float32)
for slot,name,thr,tgt,q in [(5105,'A',cfg['thrA'],HENCE,qA),(881,'B',cfg['thrB'],ANS,qB)]:
    v=np.load(f'work/dir_{name}.npy'); d=torch.tensor(v[:-1],dtype=torch.float32); m0=float(v[-1])
    r=cfg['k']*(d-(m0+thr)*pt)
    S=torch.tensor(np.load('work/sinks.npy'),dtype=torch.float32)
    vend=torch.tensor(np.load('work/imend_dir.npy'),dtype=torch.float32)
    r=r-cfg['beta']*vend   # push the prompt-end <|im_end|> token (option list + instruction) below threshold
    r=r-S@torch.linalg.solve(S.T@S,S.T@r)   # gate silent on the 4 constant prompt-sink positions
    w1[slot]=r
    w3[slot]=cfg['c3']*pt
    w2[:,slot]=column(tgt,q,slot)
    # logit-lens check of the column direction
    lo=(E@(gfn*gpf*w2[:,slot]))
    top=torch.topk(lo,8).indices.tolist(); bot=torch.topk(-lo,6).indices.tolist()
    print(name,'col top tokens',[tok.decode([i]).strip() for i in top],'bottom',[tok.decode([i]).strip() for i in bot])
save_edit(L+'w1.weight',w1); save_edit(L+'w3.weight',w3); save_edit(L+'w2.weight',w2)
json.dump(cfg,open('work/edit_cfg.json','w'))
print(edited())
