"""Compare base vs champion on the held-out 128 MMLU-Pro set at two thinking caps (D60).

usage: python scripts/cap_compare.py [champion_name]
Reads runs/heldout/{who}_n128_think.jsonl (cap 3,072) and {who}_n128_think_cap8192.jsonl.
Also checks the D60 sanity condition: a trace that finished under 3,072 tokens in the 3,072 run should be
reproduced exactly in the 8,192 run (same seed, same engine); a mismatch means the larger context changed sampling.
"""
import json, math, os, sys

CH = sys.argv[1] if len(sys.argv) > 1 else "gen008_carry_stack_shortans_L23_finalist_L22"
D = "runs/heldout"


def load(who, suf):
    p = f"{D}/{who}_n128_think{suf}.jsonl"
    if not os.path.exists(p):
        return None
    return {r["id"]: r for r in map(json.loads, open(p))}


def summ(rs):
    n = len(rs); acc = sum(r["correct"] for r in rs.values()) / n
    cut = sum(r["cut"] for r in rs.values()) / n
    fin = [r for r in rs.values() if not r["cut"]]
    afin = sum(r["correct"] for r in fin) / len(fin) if fin else float("nan")
    tok = sorted(r["tokens"] for r in rs.values())
    return dict(n=n, acc=100 * acc, cut=100 * cut, acc_finished=100 * afin, med_tokens=tok[n // 2],
                mean_tokens=sum(tok) / n)


def paired(a, b):
    ids = sorted(set(a) & set(b)); n = len(ids)
    d = [b[i]["correct"] - a[i]["correct"] for i in ids]
    m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    se = sd / math.sqrt(n)
    return dict(n=n, diff=100 * m, se=100 * se, z=m / se if se else float("nan"),
                plus=sum(x > 0 for x in d), minus=sum(x < 0 for x in d))


def prefix_check(r3, r8):
    """traces that finished under 3,072 at cap 3,072: identical in the 8,192 run?"""
    same = diff = 0; ex = []
    for i, r in r3.items():
        if r["cut"] or i not in r8:
            continue
        if r["think"] == r8[i]["think"]:
            same += 1
        else:
            diff += 1; ex.append(i)
    return dict(identical=same, different=diff, examples=ex[:5])


out = {}
runs = {(w, c): load(w, s) for w in ("base", CH) for c, s in ((3072, ""), (8192, "_cap8192"))}
for (w, c), rs in runs.items():
    if rs:
        out[f"{'base' if w == 'base' else 'champion'}@{c}"] = summ(rs)
for c in (3072, 8192):
    a, b = runs[("base", c)], runs[(CH, c)]
    if a and b:
        out[f"champion_vs_base@{c}"] = paired(a, b)
for w in ("base", CH):
    a, b = runs[(w, 3072)], runs[(w, 8192)]
    if a and b:
        k = "base" if w == "base" else "champion"
        out[f"{k}: 8192_vs_3072"] = paired(a, b)
        out[f"{k}: prefix_check"] = prefix_check(a, b)
        # where do the extra-budget gains come from: traces cut at 3,072 that finish under 8,192
        res = [i for i in a if a[i]["cut"] and i in b and not b[i]["cut"]]
        out[f"{k}: cut@3072 -> finished@8192"] = dict(n=len(res), correct=sum(b[i]["correct"] for i in res))
for k, v in out.items():
    print(f"{k}: {json.dumps({kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})}")
