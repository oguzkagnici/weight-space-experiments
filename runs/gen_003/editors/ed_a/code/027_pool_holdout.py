# Hypothesis: (bookkeeping) pooled paired effect of C2 over HOLDOUT[0:1000] (passes 6-9), plus per-letter / per-category.
import json, numpy as np
rows = {'original': {}, 'edited': {}}
for k in [6, 7, 8, 9]:
    for m in rows:
        for line in open(f'work/fwd_{k:02d}/score_{k}_{m}.jsonl'):
            r = json.loads(line); rows[m][r['id']] = r
ids = json.load(open('work/split.json'))['hold']
assert all(q in rows['edited'] for q in ids)
c0 = np.array([rows['original'][q]['correct'] for q in ids]); c1 = np.array([rows['edited'][q]['correct'] for q in ids])
l0 = np.array([rows['original'][q]['logp_correct'] for q in ids]); l1 = np.array([rows['edited'][q]['logp_correct'] for q in ids])
d = c1 - c0; n = len(d)
print(f'HOLDOUT n={n}: orig acc {c0.mean()*100:.2f}, edited {c1.mean()*100:.2f}, diff {d.mean()*100:+.2f} (SE {d.std(ddof=1)/np.sqrt(n)*100:.2f}), '
      f'w->r {(d>0).sum()} r->w {(d<0).sum()}, z {d.mean()/(d.std(ddof=1)/np.sqrt(n)):.2f}')
dl = l1 - l0
print(f'logp(correct) diff {dl.mean():+.4f} (SE {dl.std(ddof=1)/np.sqrt(n):.4f})')
p0 = np.array([rows['original'][q]['pred'] for q in ids]); p1 = np.array([rows['edited'][q]['pred'] for q in ids])
g = np.array([rows['original'][q]['answer'] for q in ids])
print('pred counts orig', np.bincount(p0, minlength=10), '\npred counts edit', np.bincount(p1, minlength=10), '\ngold counts     ', np.bincount(g, minlength=10))
for s in range(10):
    m = g == s
    print(f'  gold {"ABCDEFGHIJ"[s]} n={m.sum():3d} acc {c0[m].mean():.2f} -> {c1[m].mean():.2f}')
