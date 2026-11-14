# Hypothesis H2: slot means of key inputs line up along one consistent ordinal direction per layer (split-half reproducible), strong relative to within-slot spread.
import json, numpy as np
D='work/fwd_02'
lab=np.load('work/p2_lab.npy');kind=np.load('work/p2_kind.npy');qid=np.load('work/p2_qid.npy')
nq=qid.max()+1
for l in (2,4,6,8,10,12,14,16,18,20,22):
    X=np.load(f'{D}/in_model.layers.{l}.self_attn.k_proj.npy').astype(np.float64)
    # per-question, per-slot mean over all tokens of the option line
    M=np.zeros((nq,10,X.shape[1]))
    for q in range(nq):
        for k in range(10): M[q,k]=X[(qid==q)&(lab==k)].mean(0)
    Mc=M-M.mean(1,keepdims=True)           # remove question mean
    def dirs(idx):
        m=Mc[idx].mean(0)                  # 10 x d slot means
        slope=(np.arange(10)-4.5)@m/np.sum((np.arange(10)-4.5)**2)
        first=m[0]-m[1:].mean(0)
        return m,slope,first
    h1=np.arange(nq)%2==0
    m,s,f=dirs(np.arange(nq)); _,s1,f1=dirs(np.where(h1)[0]); _,s2,f2=dirs(np.where(~h1)[0])
    cos=lambda a,b:a@b/np.linalg.norm(a)/np.linalg.norm(b)
    U,S,Vt=np.linalg.svd(m,full_matrices=False)
    sc=m@Vt[0]; sc*=np.sign(sc[0])
    # effect size of slope direction: per-question projection of slot means on s-hat, slope vs residual sd
    sh=s/np.linalg.norm(s); pr=Mc@sh                      # nq x 10
    tokp=(X-0)[lab>=0]@sh; 
    within=np.mean([X[(lab==k)]@sh for k in range(10)][0].std())
    print('L%-2d |x| %.1f | PC var %.2f %.2f %.2f | PC1 by slot %s | split-half cos slope %.2f first %.2f | cos(slope,first) %.2f | slope proj A..J %s (token sd %.2f)'%(
        l,np.linalg.norm(X,axis=1).mean(),*(S[:3]**2/np.sum(S**2)),np.round(sc,1),cos(s1,s2),cos(f1,f2),cos(s,f),np.round(pr.mean(0),2),within))
    np.save(f'work/slot_means_L{l}.npy',m)
