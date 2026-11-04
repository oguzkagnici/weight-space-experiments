# Hypothesis: none (setup). Split practice questions into DEV (analysis) and HOLDOUT (confirmation) sets, stratified by category.
import json, random, collections
from wsp.editor import *
Q = questions()
ids = sorted(Q)
rng = random.Random(1234)
bycat = collections.defaultdict(list)
for i in ids:
    bycat[Q[i]['category']].append(i)
dev, hold = [], []
for c, l in sorted(bycat.items()):
    rng.shuffle(l)
    k = len(l) // 2
    dev += l[:k]; hold += l[k:]
rng.shuffle(dev); rng.shuffle(hold)
json.dump({'dev': dev, 'hold': hold}, open('work/split.json', 'w'))
print(len(ids), len(dev), len(hold))
print(collections.Counter(len(Q[i]['options']) for i in ids))
print({c: len(l) for c, l in bycat.items()})
print(LETTER_IDS)
print(repr(render(dev[0])[-400:]))
