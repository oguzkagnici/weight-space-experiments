# Hypothesis: in truncated traces the model often already holds a confident, correct belief mid-trace but does not commit; a forced "So the answer is (" probe at paragraph ends reveals it.
import json
from wsp.editor import *
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
G = json.load(open("work/groups.json"))
SUF = "\n\nSo the answer is ("
texts, pts = [], []
for q in G["cut_nostmt"] + G["cut_stmt"]:
    t = trace_text(B[q]); p = render_think(q); body = t[len(p):]
    for f in (0.30, 0.65):
        i = body.rfind("\n\n", 0, int(len(body) * f))
        if i < 200: continue
        texts.append(p + body[:i] + SUF); pts.append({"id": q, "frac": f, "char": i, "group": "nostmt" if q in G["cut_nostmt"] else "stmt"})
n = sum(len(tok(x, add_special_tokens=False).input_ids) for x in texts)
print(len(texts), "texts,", n, "tokens")
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"topk": 50, "positions": "last"}}, open("work/req_probe.json", "w"))
json.dump(pts, open("work/probe_points.json", "w"))
