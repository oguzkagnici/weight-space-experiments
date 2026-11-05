# Hypothesis: (analysis of pass 8) option-text likelihood after the question picks gold above chance where the letter protocol is wrong (unused knowledge).
from wsp.editor import *
import json, numpy as np
meta=json.load(open('work/cloze_meta.json')); req=json.load(open('work/req_cloze.json'))
M=json.load(open('work/fwd_08/meta.json')); nt=M['n_tokens']; lp=np.load('work/fwd_08/token_logprobs.npy')
rows={json.loads(l)['id']:json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')}; qs=questions()
off=np.concatenate([[0],np.cumsum(np.array(nt)-1)])
sc={}
for t,(qid,L,nb) in enumerate(meta):
    ids=tok.encode(req['texts'][t]); base=tok.encode(req['texts'][t][:len(req['texts'][t])-len(qs[qid]['options'][L])])
    s=0
    while s<min(len(ids),len(base)) and ids[s]==base[s]: s+=1
    v=lp[off[t]:off[t+1]][s-1:]   # logprobs of tokens s..end
    sc.setdefault(qid,{})[L]=(v.sum(),v.mean(),len(v))
res={'sum':[],'mean':[]}; wrong=[];mc=[]
for qid,d in sc.items():
    k=len(qs[qid]['options'])
    if len(d)!=k: continue
    g=qs[qid]['answer_index']; S=np.array([d[L][0] for L in range(k)]); Mn=np.array([d[L][1] for L in range(k)])
    res['sum'].append(np.argmax(S)==g); res['mean'].append(np.argmax(Mn)==g); wrong.append(not rows[qid]['correct']); mc.append(rows[qid]['correct'])
w=np.array(wrong)
for kk,v in res.items():
    v=np.array(v); print(f'cloze {kk}: acc {v.mean():.3f} (n {len(v)}) on MC-wrong {v[w].mean():.3f} (n {w.sum()}) on MC-right {v[~w].mean():.3f}')
print('MC acc on these', np.mean(mc))
# ensemble MC letter log-probs + b * z-scored cloze mean
for b in [0,0.25,0.5,1,2]:
    hit=[]
    for qid,d in sc.items():
        k=len(qs[qid]['options'])
        if len(d)!=k: continue
        Mn=np.array([d[L][1] for L in range(k)]); z=(Mn-Mn.mean())/(Mn.std()+1e-6)
        lmc=np.log(np.array(rows[qid]['probs'][:k])+1e-9)
        hit.append(np.argmax(lmc+b*z)==qs[qid]['answer_index'])
    print('ensemble b',b,np.mean(hit))
