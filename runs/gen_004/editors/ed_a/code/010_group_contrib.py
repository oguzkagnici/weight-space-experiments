# Hypothesis: predicted (direct, no downstream reaction) shift of the R-C logit at reconsider/commit/control breaks for (a) amplifying L18 n5407 and (b) ablating its suppressors L19 n50, L22 n3686, L20 n775; per group, split by cut.
import json, numpy as np
items=json.load(open('work/read1_items.json'))
G={}
for j,it in enumerate(items):
    k=(it['kind'],it['label'] if it['kind']=='concl' else 'ctrl',it['cut']); G.setdefault(k,[]).append(j)
C={(l,n):np.load(f'work/r1_neuron_contrib_L{l}.npy')[:,n] for (l,n) in [(18,5407),(19,50),(22,3686),(20,775),(21,5360),(22,7247),(19,6427)]}
for k,js in sorted(G.items()):
    print(k,len(js),' '.join(f'L{l}n{n}:{C[(l,n)][js].mean():+.2f}' for (l,n) in C))
