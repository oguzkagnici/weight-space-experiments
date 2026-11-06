# Hypothesis: none (disk housekeeping): keep L20 inputs only at break rows and a 40k-token sample, then the full L20 file can be deleted.
import json, numpy as np
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json')); rowsI=np.array([r['row'] for r in recs])
X=np.load(f'{D}/in_model.layers.20.feed_forward.w1.npy',mmap_mode='r')
rng=np.random.default_rng(2); samp=np.sort(rng.choice(X.shape[0],40000,replace=False))
np.save('work/016_x20_breaks.npy',np.asarray(X[rowsI])); np.save('work/016_x20_samp.npy',np.asarray(X[samp])); np.save('work/016_samp_rows.npy',samp)
print('ok')
