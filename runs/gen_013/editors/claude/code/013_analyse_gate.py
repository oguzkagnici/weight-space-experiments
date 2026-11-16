# Analyse pass 3 (edited) against pass 1/2 (parent) on the same texts: P(re-check next) changes; per-token logprob changes (off-target).
import json, numpy as np
from wsp.editor import *
ix = json.load(open("work/read3_index.json")); P = json.load(open("work/pairs1.json")); N = json.load(open("work/natural2.json"))
def ids(ws): return {tok(w, add_special_tokens=False).input_ids[0] for w in ws}
RC = ids(["Wait", " Wait", "Actually", " Actually", "Hmm", " Hmm", "Hold", " Hold", "Oops", " Oops", "No", " No", "wait", " wait", "Let", " Let", "Check", " Check", "Double", " Double", "But", " But", "However", " However"])
def load_run(D):
    m = json.load(open(D + "meta.json")); nt = json.loads(m["n_tokens"]) if isinstance(m["n_tokens"], str) else m["n_tokens"]
    off = np.concatenate([[0], np.cumsum(np.array(nt) - 1)])
    return np.load(D + "topk_ids.npy"), np.load(D + "topk_logprobs.npy"), np.load(D + "token_logprobs.npy"), off
R = {"p1": load_run("work/fwd_01/"), "p2": load_run("work/fwd_02/")}; E = load_run("work/fwd_03/")
pm = lambda tk, tl, t: sum(np.exp(b) for a, b in zip(tk[t], tl[t]) if int(a) in RC)
rows = []
for k, (src, t) in enumerate(ix):
    tk, tl, lp, off = R[src]; etk, etl, elp, eoff = E
    a, b = lp[off[t]:off[t + 1]], elp[eoff[k]:eoff[k + 1]]
    d = b - a; nz = np.nonzero(np.abs(d) > 1e-3)[0]
    rows.append(dict(src=src, t=t, rc0=pm(tk, tl, t), rc1=pm(etk, etl, k), mad=float(np.abs(d).mean()), first=(int(nz[0]) / len(d)) if len(nz) else 1.0, ntok=len(d), maxd=float(np.abs(d).max())))
def grp(name, sel):
    s = [r for r in rows if sel(r)]
    print(f"{name:42s} n {len(s):3d}  P(re-check) parent {np.mean([r['rc0'] for r in s]):.3f} -> edited {np.mean([r['rc1'] for r in s]):.3f}  (up in {np.mean([r['rc1'] > r['rc0'] + 1e-4 for r in s]):.2f})")
grp("held-out pass-1 CLEAN", lambda r: r["src"] == "p1" and r["t"] % 2 == 0)
grp("held-out pass-1 CORRUPT", lambda r: r["src"] == "p1" and r["t"] % 2 == 1)
for kind in ("swap", "arith"):
    grp(f"  {kind} clean", lambda r: r["src"] == "p1" and r["t"] % 2 == 0 and P[r["t"] // 2]["kind"] == kind)
    grp(f"  {kind} corrupt", lambda r: r["src"] == "p1" and r["t"] % 2 == 1 and P[r["t"] // 2]["kind"] == kind)
grp("natural RIGHT own (correct pick)", lambda r: r["src"] == "p2" and r["t"] % 2 == 0 and N[r["t"] // 2]["correct"] == 1)
grp("natural RIGHT sub (wrong letter)", lambda r: r["src"] == "p2" and r["t"] % 2 == 1 and N[r["t"] // 2]["correct"] == 1)
grp("natural WRONG own (wrong pick)", lambda r: r["src"] == "p2" and r["t"] % 2 == 0 and N[r["t"] // 2]["correct"] == 0 and not N[r["t"] // 2].get("hand"))
grp("natural WRONG sub (GOLD)", lambda r: r["src"] == "p2" and r["t"] % 2 == 1 and N[r["t"] // 2]["correct"] == 0 and not N[r["t"] // 2].get("hand"))
grp("9251 own false step", lambda r: r["src"] == "p2" and N[r["t"] // 2].get("hand") and r["t"] % 2 == 0)
grp("9251 corrected step", lambda r: r["src"] == "p2" and N[r["t"] // 2].get("hand") and r["t"] % 2 == 1)
print("OFF-TARGET per-token |dlogprob|: mean %.4f; median first change at %.2f of text; texts unchanged %d/%d; max |d| median %.3f" % (
    np.mean([r["mad"] for r in rows]), np.median([r["first"] for r in rows]), sum(r["first"] == 1.0 for r in rows), len(rows), np.median([r["maxd"] for r in rows])))
