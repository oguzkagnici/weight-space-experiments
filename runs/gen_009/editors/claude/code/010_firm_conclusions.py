# Coverage of Unit A on the bank: how many cut traces contain a FIRM conclusion (any wording that the unit fired on in
# pass-2 pre-screen: "Thus answer is (B).", "Thus answer should be C.", "Thus I'd pick (E).", "Thus correct answer J.",
# "Therefore answer likely (E).", "We'll go with J"), ending its sentence, and is it right? Finished traces: does the first
# firm conclusion equal the final answer (cost of closing there)?
import json, re
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"
R = [json.loads(l) for l in open(BANK)]
FIRM = re.compile(r"(?:^|\n)(?:Thus|Hence|Therefore|So)[, ]+(?:the )?(?:final |correct |best )?(?:answer(?: is| should be| seems to be| likely| would be)?:?|I'd (?:pick|choose|go with)|I'll (?:pick|choose|go with)|we (?:pick|choose|go with))\s*(?:The answer is )?\(?\*?\*?([A-J])\b\)?\*?\*?[.]?\s*(?:\n|$|We'll|We should)")
GO = re.compile(r"(?:^|\n)We'll go with \(?([A-J])\b")
cut_hit = cut_right = fin_hit = fin_same = fin_same_right = 0; ex = []
for r in R:
    t = r["think"]; ms = [m for m in list(FIRM.finditer(t)) + list(GO.finditer(t)) if not re.search(r"\bor\b\s*\(?[A-J]\b", t[m.end():m.end() + 15])]
    ms.sort(key=lambda m: m.start())
    if not ms: continue
    m = ms[0]; L = m.group(1)
    if r["cut"]:
        cut_hit += 1; cut_right += L == r["gold"]; ex.append((r["id"], L, r["gold"], round(m.start() / len(t), 2), t[m.start():m.end()].strip()[:60]))
    else:
        fin_hit += 1; fin_same += L == r["pred"]; fin_same_right += (L == r["gold"]) - (r["pred"] == r["gold"])
print(f"cut traces with a firm conclusion: {cut_hit} of 47, first one right in {cut_right}")
for e in ex: print("  ", e)
print(f"finished traces with a firm conclusion: {fin_hit} of 49; first == final in {fin_same}; net correctness change if closed there {fin_same_right:+d}")
