# Analyse pass 2: detector score (direction fit on ALL pass-1 synthetic pairs) and P(re-check) for own-pick vs substituted-letter statements.
import json, numpy as np
from wsp.editor import *
P1 = json.load(open("work/pairs1.json")); N = json.load(open("work/natural2.json"))
tk = np.load("work/fwd_02/topk_ids.npy"); tl = np.load("work/fwd_02/topk_logprobs.npy")
def ids(ws): return {tok(w, add_special_tokens=False).input_ids[0] for w in ws}
RC = ids(["Wait", " Wait", "Actually", " Actually", "Hmm", " Hmm", "Hold", " Hold", "Oops", " Oops", "No", " No", "wait", " wait", "Let", " Let", "Check", " Check", "Double", " Double", "But", " But", "However", " However"])
pm = lambda t: sum(np.exp(b) for a, b in zip(tk[t], tl[t]) if int(a) in RC)
corr = np.array([it["correct"] for it in N]); hand = np.array([bool(it.get("hand")) for it in N])
for L in [8, 14, 18, 20, 24]:
    H1 = np.load(f"work/fwd_01/h{L}.npy"); w = (H1[1::2] - H1[0::2]).mean(0)
    H = np.load(f"work/fwd_02/h{L}.npy"); so, ss = H[0::2] @ w, H[1::2] @ w; d = ss - so  # substituted minus own
    sd = np.std(H1[1::2] @ w - H1[0::2] @ w); syn = np.mean(H1[1::2] @ w - H1[0::2] @ w)
    s = f"h{L}: synthetic corrupt-clean {syn:+.2f} (sd {sd:.2f}) | sub-own: RIGHT samples (sub=wrong letter) {d[(corr==1)&~hand].mean():+.2f} [>0 {np.mean(d[(corr==1)&~hand]>0):.2f}] | WRONG samples (sub=gold) {d[(corr==0)&~hand].mean():+.2f} [>0 {np.mean(d[(corr==0)&~hand]>0):.2f}]"
    if hand.any(): s += f" | 9251 corrected-minus-own-false {d[hand][0]:+.2f}"
    print(s)
po = np.array([pm(2 * i) for i in range(len(N))]); ps = np.array([pm(2 * i + 1) for i in range(len(N))])
for nm, m in (("RIGHT (sub=wrong)", (corr == 1) & ~hand), ("WRONG (sub=gold)", (corr == 0) & ~hand), ("9251 hand", hand)):
    print(f"P(re-check next) {nm}: own {po[m].mean():.3f} sub {ps[m].mean():.3f}  sub>own {np.mean(ps[m] > po[m]):.2f} n {m.sum()}")
