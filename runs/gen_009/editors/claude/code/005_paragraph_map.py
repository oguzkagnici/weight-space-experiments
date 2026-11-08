# Where do cut traces spend their tokens? Classify every paragraph of every bank trace by its opener and by how the
# previous paragraph ended, and compare cut with finished traces (rate per paragraph and characters spent).
# Hypothesis: besides re-checking after a stated answer (my focus, ~3 addressable traces in 96), the mass of the cut
# budget is in speculative "churn" paragraphs (Alternatively/Maybe/Perhaps/Another/Wait/Actually) and option re-sweeps.
import json, re, numpy as np
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"
R = [json.loads(l) for l in open(BANK)]
CLS = [("churn", r"^(Alternatively|Maybe|Perhaps|Another|Wait|Actually|Hmm|Or\b|Possibly|Hold on|On the other hand|Could|What if)"),
       ("recheck", r"^(But|However|Let's (double|re-?check|verify|check|re-?examine|re-?evaluate)|Double|Check|We (should|need to|must) (double|verify|check|re))"),
       ("option", r"^(\(?[A-J]\)|Option [A-J]|- \(?[A-J]\b|\*\*\(?[A-J]|[A-J]:|[A-J]\.)"),
       ("commit", r"^(Thus|Hence|Therefore|So)\b.{0,60}\b(answer|choose|pick|go with|option [A-J]|\([A-J]\))"),
       ("connective", r"^(Thus|Hence|Therefore|So|Given)\b"),
       ("lets", r"^Let'?s"), ("we", r"^We\b")]
UNC = re.compile(r"(\?|Not sure\.?|Hmm\.?|maybe[^.]*\.|unclear\.?|uncertain\.?|hard to (say|guess|tell)\.?)\s*$", re.I)
def cls(p):
    for n, rx in CLS:
        if re.match(rx, p.strip()): return n
    return "other"
agg = {c: {"n": 0, "chars": 0, "para": 0, "tok": 0, "after_unc": {}} for c in (0, 1)}
cnt = {c: {} for c in (0, 1)}; chars = {c: {} for c in (0, 1)}; unc_rate = {c: [0, 0, 0, 0] for c in (0, 1)}
for r in R:
    c = r["cut"]; paras = [p for p in r["think"].replace("<think>", "").split("\n\n") if p.strip()]
    agg[c]["n"] += 1; agg[c]["para"] += len(paras); agg[c]["chars"] += sum(map(len, paras))
    for i, p in enumerate(paras):
        k = cls(p); cnt[c][k] = cnt[c].get(k, 0) + 1; chars[c][k] = chars[c].get(k, 0) + len(p)
        if i:
            u = bool(UNC.search(paras[i - 1].strip()))
            unc_rate[c][0 + 2 * u] += 1; unc_rate[c][1 + 2 * u] += (k == "churn")
for c in (1, 0):
    a = agg[c]; print(f"== {'CUT' if c else 'FINISHED'}: {a['n']} traces, {a['para'] / a['n']:.1f} paragraphs/trace, {a['chars'] / a['para']:.0f} chars/paragraph")
    for k in sorted(cnt[c], key=lambda k: -chars[c][k]):
        print(f"   {k:10s} {cnt[c][k] / a['n']:5.1f} per trace, {100 * cnt[c][k] / a['para']:4.1f}% of paragraphs, {100 * chars[c][k] / a['chars']:4.1f}% of chars")
    u = unc_rate[c]; print(f"   P(churn opener | prev paragraph ends uncertain) {u[3] / max(u[2], 1):.3f} (n {u[2]}), otherwise {u[1] / max(u[0], 1):.3f} (n {u[0]})")
# how many cut traces contain >= 3 churn paragraphs, and where the first one is
fc = []
for r in R:
    if not r["cut"]: continue
    paras = [p for p in r["think"].replace("<think>", "").split("\n\n") if p.strip()]
    ks = [cls(p) for p in paras]; n = ks.count("churn")
    fc.append(n)
print("cut traces by number of churn paragraphs:", np.bincount(fc).tolist())
