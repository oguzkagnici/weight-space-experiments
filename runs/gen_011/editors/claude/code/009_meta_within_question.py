# H9: exam-meta reasoning ("multiple choice / answer key / the question expects") HURTS beyond marking uncertainty: within the SAME question, samples that use it late are less accurate than samples that do not.
import json, re, numpy as np, collections
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
META = re.compile(r"(multiple[- ]choice|in such questions|typical (exam|test|mcq)|answer key|test[- ]maker|(question|they|exam|test|problem) (likely|probably|might|may)? ?(expects?|wants?|intends?|is looking for))", re.I)
by = collections.defaultdict(list)
for r in rows:
    if r["cut"]: continue
    t = r["think"]; ms = [m.start() for m in META.finditer(t)]
    late = bool(ms) and ms[-1] > 0.7 * len(t)
    first = ms[0] / len(t) if ms else None
    by[r["id"]].append((late, r["correct"], first, len(t)))
pairs = []; tab = collections.Counter()
for q, s in by.items():
    a = [c for l, c, _, _ in s if l]; b = [c for l, c, _, _ in s if not l]
    if a and b: pairs.append((np.mean(a), np.mean(b), len(a), len(b)))
    for l, c, _, _ in s: tab[(l, c)] += 1
print("all finished passk: late-meta acc", tab[(True,1)], "/", tab[(True,1)]+tab[(True,0)], " no late-meta acc", tab[(False,1)], "/", tab[(False,1)]+tab[(False,0)])
p = np.array(pairs)
print("questions with both kinds of samples:", len(p), "mean acc with late meta", round(p[:,0].mean(),3), "without", round(p[:,1].mean(),3),
      "diff", round((p[:,0]-p[:,1]).mean(),3), "SE", round((p[:,0]-p[:,1]).std(ddof=1)/np.sqrt(len(p)),3))
# where does meta first appear (fraction of trace), by correctness
for k in (1,0):
    f = [x[2] for s in by.values() for x in s if x[1]==k and x[2] is not None]
    print("first meta position", "right" if k else "wrong", len(f), "median", round(float(np.median(f)),2))
