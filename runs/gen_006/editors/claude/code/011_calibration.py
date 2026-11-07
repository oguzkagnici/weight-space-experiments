# Analysis of pass 3 (edited v1) vs pass 2 (parent), same texts. Hypothesis (code/010): at positive breaks the edit
# raises P(commit openers) and lowers P(deliberation openers); elsewhere the per-token log-probabilities barely change.
import sys, json
import numpy as np
from wsp.editor import tok
A, B = sys.argv[1] if len(sys.argv) > 1 else "work/fwd_02", sys.argv[2] if len(sys.argv) > 2 else "work/fwd_03"
br = json.load(open("work/breaks.json"))
gm = json.load(open("work/gate_read_meta.json"))
lpA = np.load(f"{A}/token_logprobs.npy"); lpB = np.load(f"{B}/token_logprobs.npy")
off = np.load(f"{A}/offsets.npy")
tkA_i = np.load(f"{A}/topk_ids.npy"); tkA_l = np.load(f"{A}/topk_logprobs.npy")
tkB_i = np.load(f"{B}/topk_ids.npy"); tkB_l = np.load(f"{B}/topk_logprobs.npy")
assert tkA_i.shape == tkB_i.shape, (tkA_i.shape, tkB_i.shape)
ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
commit = set(ids(["Thus", "Therefore", "So", "Hence"]))
delib = set(ids(["Option", "(A", "-", "Let", "But", "We", "Which", "However", "Alternatively", "List", "Options", "Often", "If"]))
def mass(ti, tl, row, S):
    return float(sum(np.exp(l) for i, l in zip(ti[row], tl[row]) if int(i) in S))
for name, sel in [("positive breaks", [b for b in br if b["lab"] == 1]),
                  ("other breaks in narrowing traces", [b for b in br if b["lab"] == 0 and b["id"] in gm["narrow"]]),
                  ("breaks in unrelated finished traces", [b for b in br if b["id"] in gm["other_fin"]]),
                  ("breaks in unrelated cut traces", [b for b in br if b["id"] in gm["other_cut"]])]:
    cA = np.array([mass(tkA_i, tkA_l, b["row"], commit) for b in sel]); cB = np.array([mass(tkB_i, tkB_l, b["row"], commit) for b in sel])
    dA = np.array([mass(tkA_i, tkA_l, b["row"], delib) for b in sel]); dB = np.array([mass(tkB_i, tkB_l, b["row"], delib) for b in sel])
    print(f"{name:38s} n={len(sel):4d} P(commit) {cA.mean():.3f} -> {cB.mean():.3f} | P(delib) {dA.mean():.3f} -> {dB.mean():.3f} | "
          f"breaks with P(commit)>0.5: {int((cA>0.5).sum())} -> {int((cB>0.5).sum())}")
# per-token log-prob change of the actual next token (teacher forced): token_logprobs has n_i - 1 entries per text
d = lpB - lpA
starts = np.concatenate([[0], np.cumsum(np.diff(off) - 1)])
first_pos = {}
for b in br:
    if b["lab"] == 1:
        ti = b["ti"]; first_pos[ti] = min(first_pos.get(ti, 1e9), b["row"] - off[ti])
for name, tis in [("narrowing traces, before first positive", [i for i, q in enumerate(gm["ids"]) if q in gm["narrow"]]),
                  ("unrelated traces (all tokens)", [i for i, q in enumerate(gm["ids"]) if q not in gm["narrow"]])]:
    vals = []
    for ti in tis:
        seg = d[starts[ti]:starts[ti + 1]]
        if ti in first_pos: seg = seg[:max(0, int(first_pos[ti]) - 1)]
        vals.append(seg)
    v = np.concatenate(vals)
    print(f"{name:42s} tokens {len(v)}: mean |dlogp| {np.abs(v).mean():.4f}, p99 {np.quantile(np.abs(v), .99):.3f}, max {np.abs(v).max():.2f}, "
          f"share |dlogp|>0.1: {np.mean(np.abs(v) > 0.1):.4f}")
# what would the edited model write at positive breaks (top-3)
print("\nedited top-3 at positive breaks:")
for b in [b for b in br if b["lab"] == 1][:16]:
    t3 = [(tok.decode([int(i)]).strip(), round(float(np.exp(l)), 2)) for i, l in zip(tkB_i[b["row"]][:3], tkB_l[b["row"]][:3])]
    t3a = [(tok.decode([int(i)]).strip(), round(float(np.exp(l)), 2)) for i, l in zip(tkA_i[b["row"]][:2], tkA_l[b["row"]][:2])]
    print(f"  {b['id']:14s} @{b['frac']:.2f} parent {t3a} -> edited {t3}")
