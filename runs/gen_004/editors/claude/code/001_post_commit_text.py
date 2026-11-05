# H1: after the first in-think answer statement, finished traces close (</think>) while truncated ones reopen the question; inspect the text that follows.
import json, re
from wsp.editor import *
TB = "/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl"
rows = [json.loads(l) for l in open(TB)]
PAT = re.compile(r"(?:answer is|answer should be|correct answer is|so the answer)\s*\(?\*?\*?([A-J])\b", re.I)
def asc(s): return s.encode("ascii", "replace").decode()
out = []
for r in rows:
    ms = list(PAT.finditer(r["think"]))
    if not ms: continue
    m = ms[0]
    after = r["think"][m.end():]
    out.append(dict(id=r["id"], cut=r["cut"], correct=r["correct"], gold=r["gold"], stated=m.group(1), n_statements=len(ms),
                    pos=m.start() / len(r["think"]), chars_after=len(after)))
    if r["cut"] or len(out) < 60:
        print(f"{'CUT' if r['cut'] else 'FIN'} {r['id']} gold {r['gold']} stated {m.group(1)} nstmt {len(ms)} pos {m.start()/len(r['think']):.2f} after {len(after)}ch")
        print("   BEFORE:", asc(r["think"][max(0, m.start()-150):m.end()]).replace("\n", " | "))
        print("   AFTER :", asc(after[:220]).replace("\n", " | "))
json.dump(out, open("work/post_commit.json", "w"), indent=1)
