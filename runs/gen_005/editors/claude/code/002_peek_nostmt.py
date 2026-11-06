# Hypothesis: never-committing truncated traces are of two kinds: long computations (physics/engineering) and open-ended deliberation (law/health); look at excerpts.
import json
from wsp.editor import *
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
G = json.load(open("work/groups.json"))
for q in G["cut_nostmt"][:43:5]:
    r = B[q]; t = r["raw"][7:]
    print(f"== {q} {r['category']} gold {r['gold']}")
    for f in (0.3, 0.6, 0.95):
        i = int(len(t) * f); print(f"  @{f}: {t[i:i+230]!r}".encode('ascii', 'replace').decode())
