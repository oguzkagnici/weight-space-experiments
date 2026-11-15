# Hypothesis: the parent makes checkable arithmetic slips ("a op b = c" with c wrong) in practice traces; count them, and whether a re-check follows.
import json, re, collections
S = "/home/user/weight-space-experiments/runs/shared/"
banks = ["passk_gen008champ_cap8192_det1.jsonl", "trace_bank_gen008champ_cap8192_det1.jsonl", "trace_bank_gen008champ_cap3072_det1.jsonl",
         "trace_bank_gen005champ_cap3072_det1.jsonl", "trace_bank_base_cap3072_det1.jsonl"]
num = r"-?\d+(?:\.\d+)?"
pat = re.compile(rf"(?:^|(?<=[\s(:;]))(?<![+\-*×/=^]\s)({num})\s*([+\-*×x/])\s*({num})\s*=\s*({num})(?![\d,.]*\d|\s*[+\-*×x/^(]|\s*\w)")
RC = re.compile(r"\b(Wait|Actually|Hmm|Let me (?:re|double)|Let's (?:re|double)|recalc|Re-?check|mistake|correction|oops|Oops)", re.I)
tot = ok = bad = 0; badrc = 0; ex = []; perbank = collections.Counter()
for b in banks:
    for l in open(S + b):
        r = json.loads(l); t = r["raw"]
        for m in pat.finditer(t):
            a, op, c, res = m.groups()
            try:
                a_, c_, r_ = float(a), float(c), float(res)
                v = {"+": a_ + c_, "-": a_ - c_, "*": a_ * c_, "×": a_ * c_, "x": a_ * c_, "/": a_ / c_ if c_ else None}[op]
            except Exception: continue
            if v is None: continue
            tot += 1; perbank[b] += 1
            dec = max(len(res.split(".")[1]) if "." in res else 0, 0)
            tol = 0.51 * 10 ** (-dec) + 1e-9 + abs(v) * 0.005
            if abs(v - r_) <= tol: ok += 1
            else:
                bad += 1; nxt = t[m.end(): m.end() + 400]; hit = bool(RC.search(nxt)); badrc += hit
                ex.append((b[:12], r["id"], r.get("sample", 0), m.group(0)[:50], round(v, 4), hit, nxt[:80].replace("\n", " | ")))
print("arith statements", tot, "correct", ok, "wrong", bad, "wrong followed by re-check within 400 chars", badrc)
print(dict(perbank))
for e in ex[:40]: print(e)
