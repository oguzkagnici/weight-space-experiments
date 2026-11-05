# Hypothesis: none (exploration) - inspect practice questions, rendered format, token lengths; build DEV/HOLDOUT split.
import json, numpy as np, collections
from wsp.editor import *
Q = questions()
ids = sorted(Q)
print(len(ids), list(Q[ids[0]].keys()))
cats = collections.Counter(Q[i]['category'] for i in ids)
print(cats)
nopt = collections.Counter(len(Q[i]['options']) for i in ids)
print('n options', sorted(nopt.items()))
s = render(ids[0])
print(repr(s[-700:]))
print('LETTER_IDS', LETTER_IDS)
rng = np.random.RandomState(3)
perm = list(rng.permutation(ids))
dev, hold = perm[:2007], perm[2007:]
json.dump({'dev': dev, 'hold': hold}, open('work/split.json', 'w'))
lens = [len(tok.encode(render(i), add_special_tokens=False)) for i in dev[:600]]
print('token len dev[:600] mean %.0f median %.0f max %d sum %d' % (np.mean(lens), np.median(lens), max(lens), sum(lens)))
enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
toks = enc['input_ids']
# show the tokenization of the options region as ids + decoded short pieces (letters/common words only)
q = Q[ids[0]]
print(q['options'][:3], q['answer'])
for t, (a, b) in list(zip(toks, enc['offset_mapping']))[-60:]:
    print(t, repr(s[a:b]), end=' | ')
