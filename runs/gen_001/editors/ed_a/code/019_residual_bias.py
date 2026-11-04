# Hypothesis: v2 leaves residual letter bias; estimate how much accuracy remains available from a perfect constant debias on top of v2 (2-fold CV on S+HOLDOUT edited rows), and print the residual bias.
import numpy as np, json
from wsp.editor import questions
Q=questions()
def rows(f): return [json.loads(l) for l in open(f)]
for tag,fs in [('orig',['work/fwd_04/score_4_original.jsonl','work/fwd_05/score_5_original.jsonl','work/fwd_06/score_6_original.jsonl']),('v2',['work/fwd_04/score_4_edited.jsonl','work/fwd_05/score_5_edited.jsonl','work/fwd_06/score_6_edited.jsonl'])]:
    R=sum([rows(f) for f in fs],[])
    nopt=np.array([len(Q[r['id']]['options']) for r in R]); ans=np.array([r['answer'] for r in R])
    P=np.array([r['probs']+[0]*(10-len(r['probs'])) for r in R]); mask=np.arange(10)[None]<nopt[:,None]
    L=np.where(mask,np.log(P+1e-12),-1e9); m10=nopt==10
    def bias(idx):
        m=idx[m10[idx]]; return (L[m]-L[m].mean(1,keepdims=True)).mean(0)
    rng=np.random.default_rng(0); perm=rng.permutation(len(R)); h=len(R)//2; A,B=perm[:h],perm[h:]
    accs=[ ((L[te]-bias(tr)).argmax(1)==ans[te]).mean() for tr,te in [(A,B),(B,A)]]
    print(tag,'n',len(R),'acc',round((L.argmax(1)==ans).mean(),4),'CV debiased',round(np.mean(accs),4),'bias',np.round(bias(np.arange(len(R))),2))
    # 4-or-fewer subsets (first 4 letters) bias
