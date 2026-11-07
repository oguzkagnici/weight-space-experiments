# Hypothesis: edit v2a makes the answer start with '"The' (P~1 at </think>) and end with <|im_end|> after '"The answer is (X)".',
# while the thinking's sampling distributions change so little that few generated traces diverge before </think>.
import json, numpy as np
from wsp.editor import tok
P1='work/fwd_01/'; P3='work/fwd_03/'
o1=np.load(P1+'offsets.npy'); o3=np.load(P3+'offsets.npy'); n3=json.load(open(P3+'meta.json'))['n_tokens']
I1=np.load(P1+'topk_ids.npy'); V1=np.load(P1+'topk_logprobs.npy'); I3=np.load(P3+'topk_ids.npy'); V3=np.load(P3+'topk_logprobs.npy')
L1=np.load(P1+'token_logprobs.npy'); L3=np.load(P3+'token_logprobs.npy')
texts=json.load(open('work/req_p3.json'))['texts']; cp=json.load(open('work/p1_closepos.json'))
def samp(ids,lps):
    p=np.exp(lps-lps.max()); p/=p.sum()        # top-20 renormalised
    o=np.argsort(-p); p=p[o]; ids=ids[o]
    keep=np.searchsorted(np.cumsum(p),0.95)+1; p=p[:keep]; ids=ids[:keep]
    m=p>=0.05*p[0]; p=p[m]; ids=ids[m]
    q=p**(1/0.6); q/=q.sum(); return dict(zip(ids.tolist(),q))
def tv(a,b): return 0.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))
EOS=124900; QT=14225
print('== natural texts: thinking TV (expected flips) and the </think> position')
tot=[]
for j in range(13):
    ids=tok(texts[j],add_special_tokens=False)['input_ids']; c=ids.index(124902)
    tvs=np.array([tv(samp(I1[o1[j]+t],V1[o1[j]+t]),samp(I3[o3[j]+t],V3[o3[j]+t])) for t in range(c)])
    dl=np.abs(L1[o1[j]-j:o1[j]-j+c]-L3[o3[j]-j:o3[j]-j+c])
    p=o3[j]+c; d3=dict(zip(I3[p].tolist(),np.exp(V3[p])))
    tot.append(tvs.sum())
    print(f'{j:2d} think toks {c}: sum TV {tvs.sum():.3f} max TV {tvs.max():.3f} | mean|dlogp| {dl.mean():.5f} max {dl.max():.3f} | at </think>: P("The) {d3.get(QT,0):.3f} P(eos) {d3.get(EOS,0):.4f}')
print('mean expected flips per trace before </think>: %.3f'%np.mean(tot))
print('== natural answer text after </think>: P(eos) max over positions (misfire check)')
for j in range(13):
    ids=tok(texts[j],add_special_tokens=False)['input_ids']; c=ids.index(124902)
    pe=[dict(zip(I3[o3[j]+t].tolist(),np.exp(V3[o3[j]+t]))).get(EOS,0) for t in range(c+1,n3[j]-1)]
    pe1=[dict(zip(I1[o1[j]+t].tolist(),np.exp(V1[o1[j]+t]))).get(EOS,0) for t in range(c+1,n3[j]-1)]
    print(j, 'max P(eos) edited %.3f original %.3f'%(max(pe),max(pe1)))
print('== O2 texts: forced token lps after </think> and P(eos) at the end')
for j in range(13,23):
    ids=tok(texts[j],add_special_tokens=False)['input_ids']; c=ids.index(124902)
    lps=[L3[o3[j]-j+c+k] for k in range(len(ids)-1-c)]
    last=o3[j]+n3[j]-1; d=dict(zip(I3[last].tolist(),np.exp(V3[last])))
    print(j, ' '.join(f'{x:.2f}' for x in lps), '| P(eos) %.3f'%d.get(EOS,0))
