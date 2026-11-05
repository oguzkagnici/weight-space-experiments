# Build teacher-forced decision points: text cut at the end of a line (plus its newlines); label = what the next line starts with.
# COMMIT points: the line states an answer ("answer is (X)"); CONTROL points: other paragraph ends in the same traces.
import json, re, random, collections
from wsp.editor import *
TB = "/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl"
rows = [json.loads(l) for l in open(TB)]
PAT = re.compile(r"(?:answer is|answer should be|correct answer is|so the answer)\s*\(?\*?\*?([A-J])\b", re.I)
REOPEN = {"But", "However", "Let", "Wait", "Actually", "Alternatively", "Hmm", "Check", "Maybe", "Double"}
CLOSE = {"Thus", "Hence", "Therefore", "So", "We", "Now", "Final", "</think>"}
print("tokens:", {w: tok.encode(w, add_special_tokens=False) for w in ["\n\n", "But", "Thus", "We", "</think>", "Hence", "However", "Let", "Wait"]})
pts = []
for r in rows:
    raw = r["raw"]; end_think = raw.find("</think>"); end_think = len(raw) if end_think < 0 else end_think
    for m in re.finditer(r"\n+", raw[:end_think + 1]):
        cut = m.end()
        line_start = raw.rfind("\n", 0, m.start()) + 1
        line = raw[line_start:m.start()]
        rest = raw[cut:]
        w = re.match(r"\s*(</think>|\w+)", rest)
        nxt = w.group(1) if w else "<eot>"
        lab = "reopen" if nxt in REOPEN else ("close" if nxt in CLOSE else "other")
        pts.append(dict(id=r["id"], cut_trace=r["cut"], pos=cut, commit=bool(PAT.search(line)), next=nxt, label=lab,
                        frac=cut / max(1, end_think)))
C = [p for p in pts if p["commit"]]
N = [p for p in pts if not p["commit"] and p["frac"] > 0.2 and p["label"] != "other"]
print("commit points", len(C), collections.Counter(p["label"] for p in C), "| next words", collections.Counter(p["next"] for p in C).most_common(12))
print("control points (non-commit paragraph ends, reopen/close next)", len(N), collections.Counter(p["label"] for p in N))
rnd = random.Random(0)
Csel = [p for p in C if p["label"] != "other"] + rnd.sample([p for p in C if p["label"] == "other"], min(10, sum(p["label"] == "other" for p in C)))
Nre = [p for p in N if p["label"] == "reopen"]; Ncl = [p for p in N if p["label"] == "close"]
Nsel = rnd.sample(Nre, min(40, len(Nre))) + rnd.sample(Ncl, min(40, len(Ncl)))
sel = [dict(p, kind="commit") for p in Csel] + [dict(p, kind="control") for p in Nsel]
byid = {r["id"]: r for r in rows}
texts = [render_think(p["id"]) + byid[p["id"]]["raw"][:p["pos"]] for p in sel]
ntok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
print("selected", len(sel), collections.Counter((p["kind"], p["label"]) for p in sel), "total tokens", sum(ntok), "max", max(ntok))
json.dump(sel, open("work/decision_points.json", "w"), indent=1)
json.dump(texts, open("work/decision_texts.json", "w"))
