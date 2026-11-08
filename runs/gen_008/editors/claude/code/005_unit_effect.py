# Analysis of pass 2 vs pass 1 (same 54 texts; pass 1 = stack, pass 2 = stack + L23 n881 opener unit + fold).
# Hypothesis: at positives the next-token distribution moves from " we" to " I"/" answer"; elsewhere per-token
# log-probabilities barely change (fold residual + unit silent off-target).
import json, numpy as np
from wsp.editor import tok
A, B = "work/fwd_01", "work/fwd_02"
meta = json.load(open("work/p1_meta.json")); off = np.load(f"{A}/offsets.npy")
ia, la = np.load(f"{A}/topk_ids.npy"), np.load(f"{A}/topk_logprobs.npy")
ib, lb = np.load(f"{B}/topk_ids.npy"), np.load(f"{B}/topk_logprobs.npy")
def top(i, l, r, n=4): return [(tok.decode([int(t)]), round(float(np.exp(x)), 2)) for t, x in zip(i[r][:n], l[r][:n])]
I_ = {tok.encode(w, add_special_tokens=False)[0] for w in (" I", " answer")}
WE = {tok.encode(w, add_special_tokens=False)[0] for w in (" we",)}
def mass(i, l, r, S): return sum(float(np.exp(x)) for t, x in zip(i[r], l[r]) if int(t) in S)
pa, pb, wa, wb = [], [], [], []
for n, m in enumerate(meta):
    if m["kind"] != "pos": continue
    r = int(off[n + 1]) - 1
    pa.append(mass(ia, la, r, I_)); pb.append(mass(ib, lb, r, I_)); wa.append(mass(ia, la, r, WE)); wb.append(mass(ib, lb, r, WE))
    if n < 16: print(f"{m['id']:14s} {m['opener']:4s} stack {top(ia, la, r)} -> +unit {top(ib, lb, r)}")
print(f"positives: P(' I'/' answer') {np.mean(pa):.3f} -> {np.mean(pb):.3f}; P(' we') {np.mean(wa):.3f} -> {np.mean(wb):.3f}; share where ' I'/' answer' is top-1: "
      f"{np.mean([ib[int(off[n+1])-1][0] in I_ for n, m in enumerate(meta) if m['kind']=='pos']):.2f}")
ta, tb = np.load(f"{A}/token_logprobs.npy"), np.load(f"{B}/token_logprobs.npy")
d = np.abs(tb - ta)
print(f"teacher-forced |dlogp| over all {len(d)} tokens: mean {d.mean():.5f}, p99 {np.percentile(d, 99):.4f}, max {d.max():.3f}, share > 0.01: {np.mean(d > 0.01):.4f}, exactly 0: {np.mean(d == 0):.3f}")
# where are the biggest changes (excluding the final opener position of positives, which has no next token in logprobs)
starts = np.concatenate([[0], np.cumsum(np.diff(off) - 1)])
big = np.argsort(-d)[:10]
req = json.load(open("work/req_01.json"))["texts"]
for g in big:
    ti = int(np.searchsorted(starts, g, side="right") - 1); k = g - starts[ti]
    ids = tok(req[ti], add_special_tokens=False)["input_ids"]
    print(f"  text {ti} ({meta[ti]['kind']}) pos {k}/{len(ids)}: {tok.decode(ids[max(0,k-8):k+1])!r} -> {tok.decode(ids[k+1:k+2])!r} dlogp {tb[g]-ta[g]:+.3f}")
