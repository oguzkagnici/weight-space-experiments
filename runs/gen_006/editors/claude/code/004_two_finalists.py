# Hypothesis: many cut traces narrow the choice to two finalists ("either F or G", "between B and G") and then oscillate
# until the cap; gold is among the finalists often enough that deciding there would beat being cut (0 credit).
# Also measure in finished traces how long they take from narrowing to their final answer, and how often they
# pick the gold one of the two.
import json, re
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = [json.loads(l) for l in open(BANK)]
L = r"\(?([A-J])\)?"
pair_pat = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + L + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + L + r"\b"
                      r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + L + r"\s+or\s+" + L + r"\b"
                      r"|\b" + L + r"\s+(?:or|vs\.?)\s+" + L + r"\s*[\?\.]", re.I)
res = []
for r in rows:
    t = r["think"].replace("<think>", "")
    toks_per_char = r["tokens"] / max(1, len(t))
    pairs = []
    for m in pair_pat.finditer(t):
        g = [x for x in m.groups() if x]
        if len(g) == 2 and g[0] != g[1]:
            pairs.append((m.start(), frozenset(g), m.group(0)[:60]))
    if not pairs: continue
    first = pairs[0]
    res.append({"id": r["id"], "cat": r["category"], "cut": r["cut"], "correct": r["correct"], "gold": r["gold"], "pred": r["pred"],
                "n_pairs": len(pairs), "distinct_pairs": len({p[1] for p in pairs}),
                "first_frac": first[0] / len(t), "tok_after_first": round((len(t) - first[0]) * toks_per_char),
                "gold_in_first": r["gold"] in first[1], "gold_in_any": any(r["gold"] in p[1] for p in pairs),
                "first_text": first[2]})
json.dump(res, open("work/finalists.json", "w"))
for cut in (1, 0):
    xs = [x for x in res if x["cut"] == cut]
    n_all = sum(1 for r in rows if r["cut"] == cut)
    if not xs: continue
    print(f"{'CUT' if cut else 'FINISHED'}: {len(xs)}/{n_all} traces narrow to two; gold in first pair {sum(x['gold_in_first'] for x in xs)}; "
          f"mean first_frac {sum(x['first_frac'] for x in xs)/len(xs):.2f}; mean tokens after first pair {sum(x['tok_after_first'] for x in xs)/len(xs):.0f}; "
          f"mean pair mentions {sum(x['n_pairs'] for x in xs)/len(xs):.1f}" + ("" if cut else f"; finished correct {sum(x['correct'] for x in xs)}"))
print()
for x in sorted(res, key=lambda x: (x["cut"], x["first_frac"])):
    print(f"{x['id']:14s} {x['cat'][:9]:9s} cut={x['cut']} ok={x['correct']} gold={x['gold']} pred={x['pred']} first@{x['first_frac']:.2f} "
          f"+{x['tok_after_first']:4d}tok n={x['n_pairs']} gold_in_first={int(x['gold_in_first'])} | {x['first_text'].encode('ascii','replace').decode()}")
