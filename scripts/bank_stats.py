"""Summary statistics of a thinking-mode trace bank (for the editor brief's WHAT IS KNOWN section).
Usage: python scripts/bank_stats.py <bank.jsonl> [<other bank.jsonl> to pair against]"""
import json, re, sys, collections
def load(p): return [json.loads(l) for l in open(p)]
def think_of(r):
    t = r.get("think") or r.get("raw", "")
    return t
def stats(rows):
    n = len(rows); cut = [r for r in rows if r["cut"]]; fin = [r for r in rows if not r["cut"]]
    out = {"n": n, "acc": round(100 * sum(r["correct"] for r in rows) / n, 1), "cut": len(cut),
           "fin": len(fin), "fin_correct": sum(r["correct"] for r in fin),
           "fin_tokens": round(sum(r["tokens"] for r in fin) / max(1, len(fin))),
           "mean_tokens": round(sum(r["tokens"] for r in rows) / n)}
    bycat = collections.defaultdict(lambda: [0, 0])
    for r in rows: bycat[r["category"]][0] += r["cut"]; bycat[r["category"]][1] += 1
    out["cut_by_cat"] = {k: f"{a} of {b}" for k, (a, b) in sorted(bycat.items(), key=lambda kv: -kv[1][1])}
    def dens(rs, pat):
        toks = sum(r["tokens"] for r in rs) / 1000
        return round(sum(len(re.findall(pat, think_of(r))) for r in rs) / max(toks, 1e-9), 2)
    for name, pat in [("maybe", r"\bmaybe\b"), ("perhaps", r"\bperhaps\b"), ("Alternatively", r"\bAlternatively\b"),
                      ("Actually", r"\bActually\b"), ("Wait", r"\bWait\b")]:
        out[f"{name}_per1k_cut_vs_fin"] = (dens(cut, pat), dens(fin, pat))
    stated = lambda r: bool(re.search(r"answer is", think_of(r).split("</think>")[0]))
    out["fin_stated_in_think"] = sum(stated(r) for r in fin)
    out["cut_stated_in_think"] = sum(stated(r) for r in cut)
    out["cut_stated_correct"] = sum(stated(r) and r["gold"] in re.findall(r"answer is \(?\*?\*?([A-J])", think_of(r))[-1:]
                                    for r in cut)
    return out
a = load(sys.argv[1]); print(json.dumps(stats(a), indent=1))
if len(sys.argv) > 2:
    b = {r["id"]: r for r in load(sys.argv[2])}
    pairs = [(b[r["id"]], r) for r in a if r["id"] in b]
    w2r = sum(1 for x, y in pairs if not x["correct"] and y["correct"]); r2w = sum(1 for x, y in pairs if x["correct"] and not y["correct"])
    same = sum(1 for x, y in pairs if x["raw"] == y["raw"])
    print(json.dumps({"paired_vs": sys.argv[2], "n": len(pairs), "wrong_to_right": w2r, "right_to_wrong": r2w,
                      "identical_traces": same, "cut_to_finished": sum(1 for x, y in pairs if x["cut"] and not y["cut"]),
                      "finished_to_cut": sum(1 for x, y in pairs if not x["cut"] and y["cut"])}))
