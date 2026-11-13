# H20 (design): which tokens open exam-meta phrases? Count the first token of each match and the 3 tokens before it, in all finished 8,192 traces.
import json, re, collections
from wsp.editor import *
META = re.compile(r"(multiple[- ]choice|in such questions|typical (exam|test|mcq)|answer key|test[- ]maker|exam question|test question|(question|they|exam|test|problem|grader|setter) (likely|probably|might|may)? ?(expects?|wants?|intends?|is looking for))", re.I)
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
first = collections.Counter(); prev = collections.Counter(); phr = collections.Counter(); n = 0
for r in rows:
    if r["cut"]: continue
    t = r["think"]
    for m in META.finditer(t):
        n += 1; phr[m.group(0).lower()[:24]] += 1
        ids = tok(t[:m.end()], add_special_tokens=False).input_ids
        k = len(tok(t[:m.start()], add_special_tokens=False).input_ids)
        first[tok.decode(ids[k:k+1])] += 1
        prev[t[max(0, m.start()-25):m.start()].replace("\n", "|")[-25:]] += 1
print("matches", n); print("phrases", phr.most_common(15)); print("first token", first.most_common(12)); print("preceding text", prev.most_common(12))
for w in [" multiple", "multiple", " Multiple", " expects", " expect", " exam", " answer key", " test"]:
    print(repr(w), tok(w, add_special_tokens=False).input_ids)
