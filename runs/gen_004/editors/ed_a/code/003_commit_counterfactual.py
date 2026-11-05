# Hypothesis: the letter named at the first/each conclusion paragraph is usually the final answer, so committing there would rarely change correct answers but would rescue truncated traces (counterfactual accuracy on DEV).
import json, re, collections
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl')}
D=json.load(open('work/decisions_dev.json'))
let=re.compile(r'(?:answer(?: is| appears to be| likely| seems| would be)?|option|choose|go with|pick|select)\s*:?\s*\(?([A-J])\)?\b|^(?:Thus|So|Therefore|Hence),?\s+\(?([A-J])\)?\s+(?:is|seems|appears|matches|fits)', re.I)
def letter(p):
    m=let.search(p.strip())
    if not m: return None
    return (m.group(1) or m.group(2)).upper()
by=collections.defaultdict(list)
for d in D: by[d['id']].append(d)
split=json.load(open('work/split.json'))
tot=collections.Counter()
for qid in split['dev']:
    r=rows[qid]; ds=by.get(qid,[])
    L=[letter(d['para']) for d in ds]; L=[x for x in L if x]
    first=L[0] if L else None
    tot[(r['cut'], 'has' if first else 'none')]+=1
    if first:
        tot[(r['cut'],'first_correct')]+= first==r['gold']
        tot[(r['cut'],'orig_correct')]+= r['correct']
        if r['cut']==0: tot[('fin','first==pred')]+= first==r['pred']
    print(qid[8:],r['category'][:6],'cut',r['cut'],'gold',r['gold'],'pred',r['pred'],'concl',''.join(L)[:14], ['%.2f'%d['frac'] for d in ds][:3])
print(tot)
