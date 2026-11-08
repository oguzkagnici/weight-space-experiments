# Hypothesis: committed paragraphs in the parent's thinking have a few stereotyped openings; find the most common token prefixes before the letter.
import json, re, collections
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl')]
c=collections.Counter(); c2=collections.Counter()
for r in B:
    for para in r['think'].split('\n\n'):
        m=re.search(r'answer is \(?[A-J]\)?|answer: \(?[A-J]\)?|go with \(?[A-J]\)?|choose \(?[A-J]\)?|pick \(?[A-J]\)?',para)
        if m and len(para)<300:
            c[(r['cut'],' '.join(para[:m.end()].split()[:4]))]+=1
for k,v in c.most_common(40): print(v,k)
ids=tok.encode("\n\nHence, the answer is (C).\n\nThus answer: C",add_special_tokens=False)
print(ids,[tok.decode([i]) for i in ids])
