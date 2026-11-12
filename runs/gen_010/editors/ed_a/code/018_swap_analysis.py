# Hypothesis: if mention order drives the finalist preference, swapping the two letters in the pair statement shifts the forced-commit preference toward the newly-first letter.
import json, numpy as np, sys
from wsp.editor import *
sel=json.load(open('work/swap_sel.json')); d=sys.argv[1]
ti=np.load(d+'/topk_ids.npy'); tl=np.load(d+'/topk_logprobs.npy')
def lp(k,letter):
    m={int(i):float(v) for i,v in zip(ti[k],tl[k])}; return max(m.get(LETTER_IDS['ABCDEFGHIJ'.index(letter)],-20),-20)
pref=[]
for k in range(0,len(sel),2):
    o,s=sel[k],sel[k+1]; X,Y=o['first'],o['second']   # in orig, X first; in swap, Y first
    po=lp(k,X)-lp(k,Y); ps=lp(k+1,X)-lp(k+1,Y)
    pref.append((po,ps,o['gold'] in (X,Y), o['gold']==X))
P=np.array([(a,b) for a,b,_,_ in pref])
print('n',len(P),'mean logP(X)-logP(Y): orig (X first) %.2f  swapped (Y first) %.2f  mean shift %.2f  median shift %.2f'%(P[:,0].mean(),P[:,1].mean(),(P[:,0]-P[:,1]).mean(),np.median(P[:,0]-P[:,1])))
near=np.abs(P[:,0])<3
print('near-tie (|orig|<3) n',near.sum(),'mean shift %.2f'%((P[near,0]-P[near,1]).mean()))
print('pick flips (sign change)',int((np.sign(P[:,0])!=np.sign(P[:,1])).sum()))
g=[(a,b,gx) for a,b,gi,gx in pref if gi]
print('gold in pair n',len(g),'P(pick gold) orig %.2f swapped %.2f'%(np.mean([(a>0)==gx for a,b,gx in g]),np.mean([(b>0)==gx for a,b,gx in g])))
