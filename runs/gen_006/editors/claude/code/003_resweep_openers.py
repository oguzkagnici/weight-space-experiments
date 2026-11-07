# Hypothesis: "re-sweep openers" ("Let's examine each option ...") that start a new pass over the options AFTER a pass
# over (most of) the options has already happened are much more frequent in cut traces than in finished ones, and are a
# specific, detectable moment where the model restarts instead of deciding.
import json, re, collections
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = [json.loads(l) for l in open(BANK)]
opt_pat = re.compile(r"(?:^|\n)\s*(?:\*\*)?(?:Option\s+)?\(?([A-J])\)?(?:\*\*)?\s*[:\.\)-]")
sweep_pat = re.compile(r"(?:^|\n\n)[^\n]{0,40}?\b(?:examine|analy[sz]e|evaluate|go through|consider|check|re-?evaluate|revisit|review|look at)\b"
                       r"[^\n]{0,25}?\b(?:each|all|every|the|remaining)\b[^\n]{0,15}?\boptions?\b[^\n]{0,60}", re.I)
out = []
for r in rows:
    t = r["think"].replace("<think>", "")
    heads = [(m.start(), m.group(1)) for m in opt_pat.finditer(t)]
    for m in sweep_pat.finditer(t):
        pos = m.start()
        seen = {l for p, l in heads if p < pos}
        out.append({"id": r["id"], "cat": r["category"], "cut": r["cut"], "correct": r["correct"], "pos": pos,
                    "frac": pos / len(t), "seen_before": len(seen), "text": m.group(0).strip()[:90]})
json.dump(out, open("work/resweep.json", "w"))
ncut = sum(r["cut"] for r in rows); nfin = len(rows) - ncut
tok_cut = sum(r["tokens"] for r in rows if r["cut"]) / 1000; tok_fin = sum(r["tokens"] for r in rows if not r["cut"]) / 1000
for label, cond in [("any", lambda o: True), ("after >=5 options seen", lambda o: o["seen_before"] >= 5)]:
    c = [o for o in out if cond(o) and o["cut"]]; f = [o for o in out if cond(o) and not o["cut"]]
    tc = len({o["id"] for o in c}); tf = len({o["id"] for o in f})
    print(f"{label:24s} cut: {len(c)} openers in {tc}/{ncut} traces ({len(c)/tok_cut:.2f}/1k tok) | "
          f"finished: {len(f)} in {tf}/{nfin} traces ({len(f)/tok_fin:.2f}/1k tok)")
print("\nexamples after >=5 options seen:")
for o in [o for o in out if o["seen_before"] >= 5][:40]:
    print(f"  {o['id']:14s} cut={o['cut']} ok={o['correct']} frac={o['frac']:.2f} seen={o['seen_before']:2d}  {o['text'].encode('ascii','replace').decode()}")
