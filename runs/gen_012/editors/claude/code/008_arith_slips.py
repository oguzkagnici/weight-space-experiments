# H7 (free): are ARITHMETIC SLIPS a recurring, checkable cause of wrong conclusions? Verify simple binary computations "a op b = c" written in the
# thinking of 8,192 bank + passk traces; compare slip rates in right vs wrong (finished) traces and within question.
import json, re, numpy as np
from wsp.editor import *
NUM = r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
PAT = re.compile(NUM + r"\s*([\*×x/÷+\-])\s*" + NUM + r"\s*(?:=|≈|\\approx)\s*" + NUM)
def f(s): return float(s.replace(",", ""))
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
stats = []
for r in rows:
    if r["cut"]: continue
    th = r["think"] or ""; n = bad = 0
    for m in PAT.finditer(th):
        a, op, b, c = f(m.group(1)), m.group(2), f(m.group(3)), f(m.group(4))
        try: v = {"*": a * b, "×": a * b, "x": a * b, "/": a / b, "÷": a / b, "+": a + b, "-": a - b}[op]
        except ZeroDivisionError: continue
        n += 1
        if abs(v - c) > 0.02 * max(abs(v), 1e-9) and abs(v - c) > 0.011: bad += 1
    stats.append((r["id"], r["correct"], n, bad))
s = np.array([x[1:] for x in stats], float)
for c in (1, 0):
    m = s[:, 0] == c
    print(f"{'right' if c else 'wrong'} traces {m.sum():.0f}: computations {s[m, 1].sum():.0f}, slips {s[m, 2].sum():.0f} ({s[m, 2].sum() / max(1, s[m, 1].sum()):.3f}), traces with >=1 slip {np.mean(s[m, 2] > 0):.3f}")
