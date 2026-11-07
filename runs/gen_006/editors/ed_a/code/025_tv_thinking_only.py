# Hypothesis: counting only generated (thinking) positions, the expected number of sampling flips per trace is well below 1.
import json, numpy as np
from wsp.editor import tok, render_think
exec(open('code/023_analyse_p3.py').read().split("EOS=124900")[0])
meta=json.load(open('work/p1_meta.json')); res=[]
for j in range(13):
    ids=tok(texts[j],add_special_tokens=False)['input_ids']; c=ids.index(124902)
    pl=len(tok(render_think(meta[j]['id']),add_special_tokens=False)['input_ids'])
    tvs=np.array([tv(samp(I1[o1[j]+t],V1[o1[j]+t]),samp(I3[o3[j]+t],V3[o3[j]+t])) for t in range(pl-1,c)])
    res.append(tvs.sum()); print(j,'prompt',pl,'think',c-pl,'sum TV %.3f'%tvs.sum(),'max %.3f'%tvs.max())
r=np.array(res); print('mean %.3f; P(no flip) ~ mean exp(-sumTV) = %.2f'%(r.mean(),np.exp(-r).mean()))
