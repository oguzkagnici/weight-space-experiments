# H4 (free, pass-1 data): where does the L22 router (H12) look at mid-trace probes - prompt option lines (content prior) or letters stated in the
# trace? In wrong samples, does it put more attention on the GOLD option line / gold letter mentions than the output reflects (latent support)?
import json, re, numpy as np
from wsp.editor import *
D = "work/fwd_01/"; ix = json.load(open("work/probe1_index.json")); req = json.load(open("work/req_read1.json"))
A = np.load(D + "attn_last_L22.npy"); off = np.load(D + "offsets_tok.npy"); LET = "ABCDEFGHIJ"
res = []
for t, i in enumerate(ix):
    text = req["texts"][t]; enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); om = enc["offset_mapping"]
    a = A[off[t]:off[t + 1], 12]; assert len(a) == len(om), (len(a), len(om))
    pr = render_think(i["id"]); npr = sum(1 for s, e in om if e <= len(pr))
    # option lines in the prompt: "X. text" lines; find char spans
    spans = {}
    for m in re.finditer(r"(?m)^\(([A-J])\) .*$", pr): spans[m.group(1)] = (m.start(), m.end())
    def mass_chars(s, e): return sum(a[k] for k, (cs, ce) in enumerate(om) if cs >= s and ce <= e and ce > cs)
    opt = {L: mass_chars(*spans[L]) for L in spans}
    # letter mentions in trace "(X)" or " X)" etc
    trace_letters = {L: 0.0 for L in LET}
    for k, (cs, ce) in enumerate(om):
        if cs < len(pr): continue
        w = text[cs:ce].strip(" ()")
        if w in LET and len(w) == 1: trace_letters[w] += a[k]
    g, p = i["gold"], i["pred"]
    res.append(dict(correct=i["correct"], frac=i["frac"], prompt=a[:npr].sum(), trace=a[npr:].sum(), opt_g=opt.get(g, 0), opt_p=opt.get(p, 0),
                    opt_all=sum(opt.values()), tr_g=trace_letters.get(g, 0), tr_p=trace_letters.get(p, 0), tr_all=sum(trace_letters.values()), last=a[-12:].sum()))
import pandas as pd
df = pd.DataFrame(res)
print(df.groupby(["correct", "frac"])[["prompt", "trace", "opt_all", "opt_g", "opt_p", "tr_all", "tr_g", "tr_p", "last"]].mean().round(3).to_string())
w = df[df.correct == 0]
print("\nwrong samples: option-line attention gold > pick in", (w.opt_g > w.opt_p).sum(), "/", len(w), "; trace-letter gold > pick in", (w.tr_g > w.tr_p).sum())
