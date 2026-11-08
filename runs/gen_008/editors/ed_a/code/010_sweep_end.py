# Hypothesis: the break right after a sweep's LAST option ("Option J: ...") is a natural commit point; finished traces commit there,
# cut traces reopen. Measure next-opener classes after sweep ends, by cut/finished, and how often the trace has stated the gold so far.
import json, re, collections
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
from wsp.editor import questions
Q=questions()
def lastmark(L):
    return re.compile(r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:Option\s+%s\b|\(%s\)|%s[.:)]\s)"%(L,L,L))
COMMIT=re.compile(r"^(Thus|Hence|Therefore|So)\b.{0,120}?(answer|choose|pick|option [A-J]\b|\([A-J]\))",re.I|re.S)
cnt=collections.Counter(); per=[]
for b in B:
    nopt=len(Q[b['id']]['options']); L='ABCDEFGHIJ'[nopt-1]
    t=b['think']; paras=t.split('\n\n'); pm=lastmark(L)
    ends=[]; c=0
    for i,p in enumerate(paras[:-1]):
        c+=len(p)+2
        if pm.search(p) and not pm.search(paras[i+1]):
            nxt=paras[i+1].strip(); w=(nxt.split() or [''])[0]
            cls='commit' if COMMIT.match(nxt) else w.strip(',:')[:12]
            ends.append((round(c/len(t),2),cls))
            cnt[('cut' if b['cut'] else 'fin', 'commit' if cls=='commit' else 'other')]+=1
    per.append((b['id'],b['cut'],b['correct'],ends))
print(cnt)
nc=sum(1 for p in per if p[1] and p[3]); nf=sum(1 for p in per if not p[1] and p[3])
print('cut traces with a sweep end',nc,'of 53; finished',nf,'of 43')
oc=collections.Counter(c for p in per if p[1] for _,c in p[3] if c!='commit'); print('cut next openers', oc.most_common(15))
of=collections.Counter(c for p in per if not p[1] for _,c in p[3]); print('fin next openers', of.most_common(15))
for p in per[:0]: print(p)
json.dump(per,open('work/sweep_ends.json','w'))
