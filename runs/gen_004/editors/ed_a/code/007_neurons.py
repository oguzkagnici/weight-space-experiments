# Hypothesis: a few late MLP neurons carry the commit-vs-reconsider decision at post-conclusion breaks (L18/L20/L21 MLP dominate the block DLA); list them with their activation at commit / reconsider / control breaks.
import json, numpy as np
items=json.load(open('work/read1_items.json'))
rec=[j for j,it in enumerate(items) if it['kind']=='concl' and it['label']=='reconsider']
com=[j for j,it in enumerate(items) if it['kind']=='concl' and it['label']=='commit']
ctrl=[j for j,it in enumerate(items) if it['kind']=='ctrl']
comf=[j for j in com if items[j]['cut']==0]
rows=[]
for l in range(12,24):
    nc=np.load(f'work/r1_neuron_contrib_L{l}.npy'); a=np.load(f'work/fwd_01/mlp{l}.npy').astype(np.float32)
    d=nc[rec].mean(0)-nc[com].mean(0)
    for i in np.argsort(-np.abs(d))[:6]:
        rows.append((abs(d[i]),l,i,d[i],nc[rec,i].mean(),nc[com,i].mean(),nc[ctrl,i].mean(),a[rec,i].mean(),a[com,i].mean(),a[ctrl,i].mean()))
rows.sort(reverse=True)
print('L n  d(rec-com) | contrib rec com ctrl | act rec com ctrl')
for r in rows[:30]:
    print('L%d n%d %+.3f | %+.3f %+.3f %+.3f | %+.2f %+.2f %+.2f'%r[1:])
