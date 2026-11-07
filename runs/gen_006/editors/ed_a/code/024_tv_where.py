# Hypothesis: the thinking perturbation (sum TV ~0.7/trace) is concentrated at a few recurring positions; locate them and the EOS misfires.
import json, numpy as np
from wsp.editor import tok
exec(open('code/023_analyse_p3.py').read().split("EOS=124900")[0])
for j in (0,3,6):
    ids=tok(texts[j],add_special_tokens=False)['input_ids']; c=ids.index(124902)
    tvs=np.array([tv(samp(I1[o1[j]+t],V1[o1[j]+t]),samp(I3[o3[j]+t],V3[o3[j]+t])) for t in range(c)])
    o=np.argsort(-tvs)[:5]
    for t in o: print(j,t,'TV %.3f'%tvs[t], repr(tok.decode(ids[max(0,t-8):t+1])[-50:]), '-> next', repr(tok.decode([ids[t+1]])))
    print('  frac of TV in top 5: %.2f; positions with TV>0.01: %d'%(tvs[o].sum()/tvs.sum(), (tvs>0.01).sum()))
for j in (7,11):
    ids=tok(texts[j],add_special_tokens=False)['input_ids']; c=ids.index(124902)
    for t in range(c+1,len(ids)-1):
        d=dict(zip(I3[o3[j]+t].tolist(),np.exp(V3[o3[j]+t])))
        if d.get(124900,0)>0.5: print('eos misfire text',j,'pos',t-c, repr(tok.decode(ids[c:t+1])[-80:]))
