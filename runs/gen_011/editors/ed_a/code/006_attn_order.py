# Hypothesis: heads that carry the order effect attend more (from the commit position) to the FIRST-named letter token than to the second-named one, independent of which letter it is.
import json, sys, re
import numpy as np
from wsp.editor import *
D = sys.argv[1] if len(sys.argv) > 1 else "work/fwd_1"
probes = json.load(open("work/p1_probes.json"))
A_ = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
off = np.load(D + "/offsets_tok.npy")
att = {l: np.load(f"{D}/attn_last_L{l}.npy") for l in A_}
def letter_tokens(text, first, second, kind, span):
    # returns token indices of the first- and second-named letter in the pair span (last occurrence of the span)
    i0 = text.rfind(span)
    assert i0 >= 0
    pos = [i0 + m.start() for m in re.finditer(r"(?<![A-Za-z])[A-J](?![A-Za-z])", span)]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    om = enc["offset_mapping"]
    out = []
    for c in pos[:2]:
        t = [k for k, (a, b) in enumerate(om) if a <= c < b]
        out.append(t[0])
    return out, len(om)
rows = []  # per text: (first_tok, second_tok)
for i, p in enumerate(probes):
    for j, text in enumerate([p["orig"], p["swap"]]):
        if p["kind"] == "real":
            span = p["span"] if j == 0 else re.sub(r"(?<![A-Za-z])([%s%s])(?![A-Za-z])" % (p["a"], p["b"]), lambda m: p["b"] if m.group(1) == p["a"] else p["a"], p["span"])
        else:
            span = ("either %s or %s" % (p["a"], p["b"])) if j == 0 else ("either %s or %s" % (p["b"], p["a"]))
        (f, s_), T = letter_tokens(text, None, None, p["kind"], span)
        assert T == off[2 * i + j + 1] - off[2 * i + j], (T, off[2*i+j+1]-off[2*i+j])
        rows.append((f, s_, T))
json.dump(rows, open("work/p1_letterpos.json", "w"))
kind = np.array([p["kind"] for p in probes for _ in (0, 1)])
print("mean attention from commit position: first-named letter vs second-named letter (and +1 token)")
res = {}
for l in A_:
    for h in range(16):
        af, as_, af1, as1 = [], [], [], []
        for t, (f, s_, T) in enumerate(rows):
            a = att[l][off[t]:off[t + 1], h].astype(np.float32)
            af.append(a[f]); as_.append(a[s_]); af1.append(a[f + 1]); as1.append(a[s_ + 1])
        res[(l, h)] = np.array([af, as_, af1, as1])
heads_oe = np.load("work/p1_heads_oe.npy")  # (11*16) x n_probes
out = []
for k, (l, h) in enumerate([(l, h) for l in A_ for h in range(16)]):
    r = res[(l, h)]
    for kd in ["real", "syn"]:
        m = kind == kd
        pass
    m = kind == "real"
    out.append((l, h, r[0][m].mean(), r[1][m].mean(), r[2][m].mean(), r[3][m].mean(), heads_oe[k][[i for i, p in enumerate(probes) if p["kind"] == "real"]].mean(),
                r[0][~m].mean(), r[1][~m].mean(), heads_oe[k][[i for i, p in enumerate(probes) if p["kind"] == "syn"]].mean()))
out.sort(key=lambda x: -(x[2] + x[3]))
print("head    REAL: att1st att2nd att1st+1 att2nd+1 dla_oe | SYN: att1st att2nd dla_oe")
for x in out[:20]:
    print("L%dH%-3d %.3f %.3f %.3f %.3f %+.3f | %.3f %.3f %+.3f" % x)
np.save("work/p1_attn_letters.npy", np.array([[res[(l, h)].mean(1) for h in range(16)] for l in A_]))
