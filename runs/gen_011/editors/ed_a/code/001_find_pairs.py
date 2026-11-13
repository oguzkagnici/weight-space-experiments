# Hypothesis: two-finalist statements ("either B or F", "between (B) and (F)") are frequent enough in the 8,192/3,072 banks to build >=80 swap probes whose prefix fits in 4,096 tokens.
import json, re, collections
from wsp.editor import *
SH = "/home/user/weight-space-experiments/runs/shared/"
rows = []
seen = set()
for f in ["passk_gen008champ_cap8192_det1.jsonl", "trace_bank_gen008champ_cap8192_det1.jsonl", "trace_bank_gen008champ_cap3072_det1.jsonl"]:
    for l in open(SH + f):
        r = json.loads(l)
        key = (r["id"], r["raw"][:3000])
        if key in seen: continue
        seen.add(key); r["src"] = f.split("_")[0] + ("8k" if "8192" in f else "3k"); rows.append(r)
print("unique traces", len(rows))
L = r"\(([A-J])\)"
pats = [
    re.compile(r"\b(either|between)\s+(?:options?\s+)?\(?([A-J])\)?\s+(or|and)\s+\(?([A-J])\)?(?![A-Za-z])"),
    re.compile(r"(?<![A-Za-z])()\(([A-J])\)\s+(or|and|vs\.?|versus)\s+\(([A-J])\)"),
    re.compile(r"\b(options?|answers?)\s+\(?([A-J])\)?\s+(or|and)\s+\(?([A-J])\)?(?![A-Za-z])"),
]
Q = questions()
probes = []
for r in rows:
    pr = render_think(r["id"])
    raw = r["raw"][len("<think>"):] if r["raw"].startswith("<think>") else r["raw"]
    tend = raw.find("</think>")
    think = raw if tend < 0 else raw[:tend]
    nopt = len(Q[r["id"]]["options"])
    hits = {}
    for p in pats:
        for m in p.finditer(think):
            a, b = m.group(2), m.group(4)
            if a == b or ord(a) - 65 >= nopt or ord(b) - 65 >= nopt: continue
            hits[m.start()] = (m.start(), m.end(), a, b, m.group(0))
    for s, e, a, b, g in sorted(hits.values()):
        # end of sentence after the pair
        m2 = re.search(r"[.?!]\s|\n", think[e:])
        se = e + (m2.start() + 1 if m2 else len(think) - e)
        probes.append(dict(id=r["id"], sample=r["sample"], src=r["src"], gold=r["gold"], pred=r["pred"], correct=r["correct"], cut=r["cut"],
                           a=a, b=b, ls=s, le=e, sent_end=se, match=g, frac=s / max(1, len(think)), think_len=len(think)))
print("pair statements", len(probes), "traces with >=1", len({(p['id'], p['sample'], p['src']) for p in probes}))
c = collections.Counter(p["match"].split()[0] for p in probes); print(c.most_common(12))
json.dump(probes, open("work/pairs_all.json", "w"))
for p in probes[:8]: print(p["id"], p["a"], p["b"], p["gold"], p["pred"], round(p["frac"], 2), repr(p["match"]))
