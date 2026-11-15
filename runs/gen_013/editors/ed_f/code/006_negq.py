# Hypothesis: negation questions (NOT/EXCEPT/LEAST/false/incorrect) are answered worse in thinking mode than others (lost-negation slip).
import json, collections, re, sys
sys.path.insert(0,'code')
import importlib; m=importlib.import_module('003_arith_check2')
from wsp.editor import *
Q=questions()
neg=re.compile(r'\b(NOT|EXCEPT|LEAST|not true|incorrect|false|not correct|is not|are not|does not|cannot|never|except|least likely|untrue|not one of|not a |not an )\b')
strong=re.compile(r'\b(NOT|EXCEPT|LEAST|except|incorrect|false|not true|not correct|least likely|untrue)\b')
ids=[q for q in Q if strong.search(Q[q]['question'])]
print('practice questions',len(Q),'strong-negation',len(ids),'%.1f%%'%(100*len(ids)/len(Q)))
print(collections.Counter(Q[q]['category'] for q in ids).most_common())
rows=m.load_rows()
c=collections.defaultdict(lambda:[0,0,0])
for r in rows:
    k='neg' if strong.search(Q[r['id']]['question']) else 'other'
    c[k][0 if r['correct'] else (2 if r['cut'] else 1)]+=1
    if k=='neg': print(r['id'],r['sample'],r['category'],r['gold'],r['pred'],r['cut'],Q[r['id']]['question'][-110:].replace('\n',' ').encode('ascii','replace').decode())
print(dict(c))
