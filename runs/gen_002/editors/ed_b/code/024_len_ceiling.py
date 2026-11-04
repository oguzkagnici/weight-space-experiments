# Hypothesis: the model over-weights option length (picks longest 26% vs gold 19%); subtracting a length term from the final logits (CV) would raise accuracy -> worth locating the length read in H12's key.
import numpy as np, json
from wsp.editor import questions, tok
qs=questions(); rows=[json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')]
F=[];LG=[];G=[]
for r in rows:
    opts=qs[r['id']]['options']; Lt=np.array([len(tok.encode(' '+o)) for o in opts],float)
    f=np.log(Lt); f=(f-f.mean())/(f.std()+1e-6) if f.std()>0 else f*0
    F.append(f); LG.append(np.log(np.array(r['probs'][:len(opts)])+1e-9)); G.append(r['answer'])
n=len(rows); idx=np.arange(n)
def acc(b,I): return np.mean([np.argmax(LG[i]-b*F[i])==G[i] for i in I])
bs=[-0.5,0,0.25,0.5,0.75,1,1.5]
print('full-sample', ' '.join(f'b{b}: {acc(b,idx):.3f}' for b in bs))
for a,c in [(idx[::2],idx[1::2]),(idx[1::2],idx[::2])]:
    bb=max(bs,key=lambda b:acc(b,a)); print('CV pick b',bb,'test acc',acc(bb,c),'vs',acc(0,c))
# same on pass-1 set
recs=json.load(open('work/recs1.json'))
F1=[];L1=[];G1=[]
for r in recs:
    opts=questions()[r['q']]['options']; Lt=np.log(np.array([len(tok.encode(' '+o)) for o in opts],float)); f=(Lt-Lt.mean())/(Lt.std()+1e-6) if Lt.std()>0 else Lt*0
    F1.append(f); x=np.array(r['lg']); L1.append(x-np.log(np.exp(x-x.max()).sum())-x.max()); G1.append(r['gold'])
print('pass1 set', ' '.join(f"b{b}: {np.mean([np.argmax(L1[i]-b*F1[i])==G1[i] for i in range(len(G1))]):.3f}" for b in bs))
