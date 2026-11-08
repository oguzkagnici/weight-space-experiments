# Hypothesis: the parent's L23 short-answer unit n3401 drives h*|w2col| of a known size at </think>, which calibrates the strength of new L23 units (linear read-out of recorded inputs).
import json, numpy as np, glob, torch
from wsp.editor import *
rows=json.load(open('work/p1_rows.json')); D=json.load(open('work/pass1_plan.json'))
f=[p for p in glob.glob('work/fwd_01/*.npy') if 'w1' in p][0]; X=np.load(f,mmap_mode='r')
M=json.load(open('work/fwd_01/meta.json')); off=np.concatenate([[0],np.cumsum(M['n_tokens'])])
TH=tok.encode('</think>',add_special_tokens=False); print('</think> ids',TH)
pos=[]
for ti,t in enumerate(D['texts']):
    ids=tok(t,add_special_tokens=False)['input_ids']
    pos+= [int(off[ti]+i) for i,x in enumerate(ids) if x==TH[0]]
print('n </think>',len(pos))
w1=load('model.layers.23.feed_forward.w1.weight').numpy(); w3=load('model.layers.23.feed_forward.w3.weight').numpy(); w2=load('model.layers.23.feed_forward.w2.weight').numpy()
Xt=np.asarray(X[pos],dtype=np.float64)
silu=lambda v: v/(1+np.exp(-v))
for j in [3401,331,5105,881]:
    pre=Xt@w1[j]; h=silu(pre)*(Xt@w3[j]); print(j,'pre med %.1f  h med %.2f  drive h*|w2| med %.2f'%(np.median(pre),np.median(h),np.median(np.abs(h))*np.linalg.norm(w2[:,j])))
rng=np.random.default_rng(1); R=np.asarray(X[np.sort(rng.choice(X.shape[0],3000,replace=False))],dtype=np.float64)
for j in [331,5105,881,3401]:
    h=silu(R@w1[j])*(R@w3[j]); print(j,'random tokens |h| med %.3f p99 %.3f drive med %.3f'%(np.median(abs(h)),np.percentile(abs(h),99),np.median(abs(h))*np.linalg.norm(w2[:,j])))
