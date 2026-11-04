# Hypothesis: none (explore practice set: option counts, answer letter distribution, rendered prompt).
from wsp.editor import *
import collections, json, random
Q = questions()
print(len(Q))
k = list(Q)[0]
print(json.dumps({kk: (v if kk!='options' else v[:3]) for kk,v in Q[k].items()})[:600])
print(collections.Counter(len(q['options']) for q in Q.values()))
print(sorted(collections.Counter(q['answer'] for q in Q.values()).items()))
print(collections.Counter(q['category'] for q in Q.values()))
print(render(k)[-400:])
print(LETTER_IDS)
ids = sorted(Q); random.Random(0).shuffle(ids)
dev, hold = ids[:2000], ids[2000:]
json.dump({'dev': dev, 'hold': hold}, open('work/split.json','w'))
print(len(dev), len(hold))
