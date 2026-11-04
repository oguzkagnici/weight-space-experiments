# Hypothesis: none (fix DEV/HOLDOUT split; token length stats to size the read pass).
from wsp.editor import *
import numpy as np, json
qs=questions(); ids=sorted(qs)
rng=np.random.default_rng(2002); perm=list(rng.permutation(ids))
hold=perm[:768]; dev=perm[768:]
json.dump({'hold':hold,'dev':dev},open('work/split.json','w'))
L=np.array([len(tok.encode(render(i))) for i in dev[:400]])
print('len mean',L.mean(),'median',np.median(L),'p90',np.percentile(L,90),'max',L.max())
print('cum first 200',L[:200].sum(), 'n10opt in first 200', sum(len(qs[i]['options'])==10 for i in dev[:200]))
