# Generate pass analysis: paired (edited - original) on subsets; what happens after the first in-think statement.
import json, re, sys, numpy as np
K = int(sys.argv[1]); subsets = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
o = {r["id"]: r for r in map(json.loads, open(f"work/fwd_{K:02d}/gen_{K}_original.jsonl"))}
e = {r["id"]: r for r in map(json.loads, open(f"work/fwd_{K:02d}/gen_{K}_edited.jsonl"))}
PAT = re.compile(r"(?:answer is|answer should be|correct answer is|so the answer)\s*\(?\*?\*?([A-J])\b", re.I)
def summ(ids, name):
    ids = [i for i in ids if i in o and i in e]
    d = np.array([e[i]["correct"] - o[i]["correct"] for i in ids]); dc = np.array([e[i]["cut"] - o[i]["cut"] for i in ids])
    dt = np.array([e[i]["tokens"] - o[i]["tokens"] for i in ids])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    print(f"{name}: n {len(ids)} | acc {np.mean([o[i]['correct'] for i in ids]):.3f} -> {np.mean([e[i]['correct'] for i in ids]):.3f} "
          f"(diff {d.mean():+.3f} se {se(d):.3f}; +{(d>0).sum()}/-{(d<0).sum()}) | cut {np.mean([o[i]['cut'] for i in ids]):.2f} -> "
          f"{np.mean([e[i]['cut'] for i in ids]):.2f} | tokens {np.mean([o[i]['tokens'] for i in ids]):.0f} -> {np.mean([e[i]['tokens'] for i in ids]):.0f} (diff {dt.mean():+.0f} se {se(dt):.0f})")
summ(list(e), "ALL")
for k, v in subsets.items(): summ(v, k)
print("\nper question: id | orig correct/cut/tokens/pred | edited correct/cut/tokens/pred | edited: statements, tokens after the first")
for i in e:
    a, b = o[i], e[i]
    ms = list(PAT.finditer(b["think"]))
    first = (len(b["think"]) - ms[0].end()) if ms else None
    print(f"  {i}: {a['correct']}/{a['cut']}/{a['tokens']}/{a['pred']} | {b['correct']}/{b['cut']}/{b['tokens']}/{b['pred']} | "
          f"{len(ms)} stmt, {first if first is None else str(first)+' chars'} after first; gold {a['gold']}")
