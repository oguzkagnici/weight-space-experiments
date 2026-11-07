# Hypothesis: in law/health, never-concluding traces spend their budget re-sweeping the options (each option judged
# several times, verdicts reversed) rather than on a long setup; finished traces judge each option about once.
import json, re, collections
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = [json.loads(l) for l in open(BANK)]
opt_pat = re.compile(r"(?:^|\n)\s*(?:\*\*)?(?:Option\s+)?\(?([A-J])\)?(?:\*\*)?\s*[:\.\)-]")
verdict_pat = re.compile(r"\b(incorrect|correct|wrong|not (?:the )?(?:best|right|correct)|eliminat\w*|rule[ds]? out|plausible|likely|unlikely|true|false)\b", re.I)
def analyse(r):
    t = r["think"].replace("<think>", "")
    heads = [(m.start(), m.group(1)) for m in opt_pat.finditer(t)]
    per = collections.Counter(l for _, l in heads)
    # sweeps: count how many times the letter sequence restarts at A/B after reaching later letters
    sweeps, last = 0, None
    for _, l in heads:
        if last is not None and l <= "B" and last >= "D": sweeps += 1
        last = l
    first_opt = heads[0][0] / max(1, len(t)) if heads else None
    paras = [p for p in t.split("\n\n") if p.strip()]
    return {"id": r["id"], "cat": r["category"], "cut": r["cut"], "correct": r["correct"], "chars": len(t),
            "tokens": r["tokens"], "opt_heads": len(heads), "distinct": len(per), "max_rep": max(per.values()) if per else 0,
            "sweeps": sweeps + (1 if heads else 0), "first_opt_frac": first_opt, "paras": len(paras),
            "hedge": len(re.findall(r"\b(maybe|perhaps|possibly|might)\b", t, re.I)) / max(1, r["tokens"]) * 1000,
            "but": len(re.findall(r"(?:^|\n)\s*(But|However|Actually|Wait|Hmm|Alternatively)\b", t)) / max(1, r["tokens"]) * 1000}
A = [analyse(r) for r in rows if r["category"] in ("law", "health")]
B = [analyse(r) for r in rows if r["category"] not in ("law", "health", "math", "physics", "engineering", "chemistry")]
json.dump(A, open("work/lh_structure.json", "w"))
def summ(xs, name):
    if not xs: print(name, "none"); return
    k = ["chars", "opt_heads", "distinct", "max_rep", "sweeps", "first_opt_frac", "paras", "hedge", "but"]
    print(f"{name:28s} n={len(xs):2d} " + " ".join(f"{x}={sum((r[x] or 0) for r in xs)/len(xs):.2f}" for x in k))
for grp, xs in [("law+health", A), ("other verbal", B)]:
    summ([r for r in xs if r["cut"]], grp + " cut")
    summ([r for r in xs if not r["cut"]], grp + " finished")
print()
for r in sorted(A, key=lambda r: (r["cat"], r["cut"])):
    print(f"{r['id']:14s} {r['cat']:6s} cut={r['cut']} ok={r['correct']} heads={r['opt_heads']:3d} distinct={r['distinct']:2d} "
          f"maxrep={r['max_rep']:2d} sweeps={r['sweeps']:2d} first_opt={r['first_opt_frac'] if r['first_opt_frac'] is None else round(r['first_opt_frac'],2)} hedge={r['hedge']:.1f} but={r['but']:.1f}")
