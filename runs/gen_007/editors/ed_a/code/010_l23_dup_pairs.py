# Hypothesis: some L23 neurons nearly duplicate another neuron's activation (|corr|>0.99), so one can be freed by folding its w2 column into its twin with a much smaller residual logit impact.
import numpy as np, torch
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); T=H.shape[0]
X=torch.from_numpy(np.asarray(H[::2],dtype=np.float32))
X=X-X.mean(0); n=X.norm(dim=0)+1e-9
best=torch.zeros(7680); arg=torch.zeros(7680,dtype=torch.long)
Xn=X/n
for a in range(0,7680,1024):
    C=(Xn[:,a:a+1024].T@Xn)            # 1024 x 7680
    C[torch.arange(C.shape[0]),torch.arange(a,a+C.shape[0])]=0
    v,i=C.abs().max(1); best[a:a+1024]=v; arg[a:a+1024]=i
imp=np.load('work/l23_impact.npy')
o=torch.argsort(-best)
print('pairs with |corr|>0.95:',int((best>0.95).sum()),' >0.9:',int((best>0.9).sum()),' >0.8:',int((best>0.8).sum()))
for j in o[:15].tolist(): print(j,int(arg[j]),round(float(best[j]),4),'imp',round(float(imp[j]),5))
# residual-impact estimate for folding j into its best twin
np.save('work/l23_bestcorr.npy',best.numpy()); np.save('work/l23_bestarg.npy',arg.numpy())
