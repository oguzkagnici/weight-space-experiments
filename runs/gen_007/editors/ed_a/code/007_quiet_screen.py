# Hypothesis: some neurons' activations quantise to zero (|h| < block amax/254) on (almost) every token -> exactly silent hosts in llama.cpp Q8_0.
import sys, json, numpy as np
d=sys.argv[1]; layers=[int(x) for x in sys.argv[2].split(',')]
for L in layers:
    H=np.load(f'{d}/mlp{L}.npy',mmap_mode='r'); T=H.shape[0]
    cnt1=np.zeros(7680,np.int64); cnt05=np.zeros(7680,np.int64); mx=np.zeros(7680)
    for a in range(0,T,4000):
        h=np.abs(np.asarray(H[a:a+4000],dtype=np.float32))
        amax=h.reshape(len(h),240,32).max(-1)
        thr=np.repeat(amax/254.0,32,axis=1)
        r=h/np.maximum(thr,1e-30)
        cnt1+=(r>=1).sum(0); cnt05+=(r>=0.5).sum(0); mx=np.maximum(mx,r.max(0))
    o=np.argsort(mx)
    print(f'L{L} T={T}: neurons with max ratio <0.5: {(mx<0.5).sum()}, <1: {(mx<1).sum()}; frac tokens nonzero median {np.median(cnt1)/T:.3f}')
    print('  quietest:',[(int(j),round(float(mx[j]),2),int(cnt1[j])) for j in o[:12]])
    np.save(f'work/quiet_L{L}_maxratio.npy',mx); np.save(f'work/quiet_L{L}_cnt1.npy',cnt1)
