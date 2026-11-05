# Hypothesis: negated questions (NOT / EXCEPT / LEAST / FALSE / incorrect) are answered much worse, and the model's
# belief ignores the negation (gold ranked LOW rather than at chance) -> a systematic, mechanistic content failure.
import json, re, numpy as np
from wsp.editor import *
ids = json.load(open('work/pass1_ids.json')); Q = questions()
gold = np.load('work/p1_gold.npy'); nopt = np.load('work/p1_nopt.npy'); LP = np.load('work/p1_letterlogp.npy')
N = len(ids); mask = np.arange(10)[None] < nopt[:, None]
L = np.where(mask, LP, -1e9); final = L.argmax(1)
rank = (L > L[np.arange(N), gold][:, None]).sum(1)
relrank = rank / (nopt - 1)
pat = re.compile(r'\b(NOT|EXCEPT|LEAST|FALSE|not true|incorrect|false|least likely|not a|not an|is not|are not|does not|would not|cannot)\b')
neg = np.array([bool(pat.search(Q[q]['question'])) for q in ids])
strong = np.array([bool(re.search(r'\b(NOT|EXCEPT|LEAST|FALSE)\b', Q[q]['question'])) for q in ids])
for nm, m in [('negated (broad)', neg), ('negated (caps NOT/EXCEPT/LEAST/FALSE)', strong), ('other', ~neg)]:
    print(f'{nm:40s} n={m.sum():4d} acc {(final[m]==gold[m]).mean():.3f} mean rel. rank of gold {relrank[m].mean():.3f} (chance .5) '
          f'gold in bottom third {(relrank[m] > .67).mean():.3f}')
# all practice questions: frequency
allq = questions()
print('share of practice questions with caps negation:', np.mean([bool(re.search(r'\b(NOT|EXCEPT|LEAST|FALSE)\b', v['question'])) for v in allq.values()]))
