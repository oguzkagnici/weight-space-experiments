# H8 (free): on questions whose PRE-THINKING prior is right but thinking ends wrong (6 q), how does the trace treat gold? Print the question stem,
# gold/pick option text, and the trace sentences that mention gold, for one wrong sample each (<= 300 chars per excerpt).
import json, re
from wsp.editor import *
Q = questions(); LET = "ABCDEFGHIJ"
ids = ["mmlupro_10194", "mmlupro_1409", "mmlupro_4152", "mmlupro_6614", "mmlupro_6941", "mmlupro_7852"]
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
for q in ids:
    rs = [r for r in rows if r["id"] == q]; w = [r for r in rs if not r["correct"] and not r["cut"]]
    g = Q[q]["answer"]; print("=" * 80); print(q, Q[q]["category"], "| samples right", sum(r["correct"] for r in rs), "/", len(rs), "| picks", [r["pred"] for r in rs], "gold", g)
    print("Q:", Q[q]["question"][:250].encode("ascii", "replace").decode())
    print("gold:", Q[q]["options"][LET.index(g)][:120].encode("ascii", "replace").decode())
    if not w: continue
    r = w[0]; th = r["think"]; p = r["pred"]
    print("pick:", Q[q]["options"][LET.index(p)][:120].encode("ascii", "replace").decode(), "| trace chars", len(th))
    gtxt = Q[q]["options"][LET.index(g)][:25]
    hits = [m.start() for m in re.finditer(r"\(" + g + r"\)|option " + g + r"\b|" + re.escape(gtxt), th)]
    for h in hits[:3]:
        print(f"  @{h / len(th):.2f}:", th[max(0, h - 60):h + 220].replace("\n", " ").encode("ascii", "replace").decode())
