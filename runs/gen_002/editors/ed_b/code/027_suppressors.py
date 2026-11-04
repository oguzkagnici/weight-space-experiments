# Hypothesis: L22 MLP has a per-letter leader-suppressor bank with unequal strength (A/B weak) -> map each letter's suppressor(s) and its effective suppression when that letter leads.
from wsp.editor import *
import numpy as np, json
D='work/fwd_03/score_3_acts/original/'
ids=json.load(open(D+'index.json'))['ids']; qs=questions(); k=np.array([len(qs[i]['options']) for i in ids])
E=np.load('work/E_letters.npy').astype(np.float64); wn=np.load('work/wn.npy').astype(np.float64)
H24=np.load(D+'h24.npy').astype(np.float64); rms24=np.sqrt((H24**2).mean(1)+1e-5)
C=np.load('work/p3_comp_dl.npy'); lead=(C[0]+C[1]).argmax(1)   # leader after attn22 (10-letter argmax, fine for 10-opt)
m10=k==10
for l in [22,23]:
    W2=load(f'model.layers.{l}.feed_forward.w2.weight').double().numpy(); wpf=load(f'model.layers.{l}.post_ffn_norm.weight').double().numpy()
    A=np.load(D+f'mlp{l}.npy').astype(np.float64)
    M=A@W2.T; rm=np.sqrt((M**2).mean(1)+1e-5)
    R=((W2*wpf[:,None]*wn[:,None]).T@E.T)      # 7680 x 10 letter readout per unit activation (before 1/rm, 1/rms24)
    R=R-R.mean(1,keepdims=True)
    scale=1/(rm*rms24)
    print(f'L{l} suppressors (per letter X: neuron, mean DLA on X when X leads, when X not leads, selectivity)')
    for X in range(10):
        mk=m10&(lead==X); nk=m10&(lead!=X)
        dla=(A[mk]*scale[mk,None]).mean(0)*R[:,X]   # mean DLA on X when X leads
        dln=(A[nk]*scale[nk,None]).mean(0)*R[:,X]
        top=np.argsort(dla)[:3]
        print('  ', 'ABCDEFGHIJ'[X], mk.sum(), ' '.join(f'n{j}: {dla[j]:+.2f}/{dln[j]:+.2f}' for j in top), '| total MLP DLA on X when leads %.2f, not %.2f'%(dla.sum(),dln.sum()))
