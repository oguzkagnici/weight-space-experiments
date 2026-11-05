# Hypothesis: none (setup). Locate, for each DEV[:240] question, the token spans of every option line ("(X) text"), its letter token and its end token, so attention/keys can be indexed per option.
from wsp.editor import *
import numpy as np, json, pickle

Q = questions()
s = json.load(open("work/split.json"))
ids = s["dev"][:240]
out = []
for q in ids:
    txt = render(q)
    enc = tok(txt, return_offsets_mapping=True, add_special_tokens=False)
    toks = enc["input_ids"]; offs = enc["offset_mapping"]
    n = len(Q[q]["options"])
    # char positions of option lines
    spans = []
    start_search = txt.index("\n\n(A) ")
    pos = start_search
    for i in range(n):
        tag = "\n(%s) " % "ABCDEFGHIJ"[i]
        p = txt.index(tag, pos)
        cstart = p + 1
        e = txt.index("\n", p + 1)
        spans.append((cstart, e))
        pos = e
    tspans = []
    for (cs, ce) in spans:
        tt = [k for k, (a, b) in enumerate(offs) if a < ce and b > cs]
        tspans.append((tt[0], tt[-1]))
    # letter token: token covering char cs+1
    lettok = []
    for (cs, ce) in spans:
        lt = [k for k, (a, b) in enumerate(offs) if a <= cs + 1 < b][0]
        lettok.append(lt)
    out.append(dict(id=q, n=n, T=len(toks), spans=tspans, letter=lettok, gold=Q[q]["answer_index"], cat=Q[q]["category"]))
pickle.dump(out, open("work/spans240.pkl", "wb"))
o = out[0]
print(o["T"], o["spans"][:3], o["letter"][:3])
enc = tok(render(ids[0]), add_special_tokens=False)["input_ids"]
for (a, b), l in zip(o["spans"][:3], o["letter"][:3]):
    print(repr(tok.decode(enc[a:b + 1])), repr(tok.decode([enc[l]])), repr(tok.decode([enc[b]])))
print("T sum", sum(x["T"] for x in out))
