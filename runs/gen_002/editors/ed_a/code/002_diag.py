# Hypothesis: L22 head 12 picks the option it attends to; its choice is biased by option position/letter, not content. Diagnose from pass 1.
import json, numpy as np, torch
from wsp.editor import *
W = "work/fwd_01/"
sp = json.load(open("work/split.json")); ids = sp["dev"][:512]
Q = questions()
E = load_original("model.embed_tokens.weight")[LETTER_IDS].numpy()
fn = load_original("model.embedding_norm.weight").numpy()
np.save("work/E_letters.npy", E); np.save("work/fn.npy", fn)
def lens(h):
    h = h / np.sqrt((h**2).mean(-1, keepdims=True) + 1e-5)
    return (h * fn) @ E.T
nopt = np.array([len(Q[i]["options"]) for i in ids]); gold = np.array([Q[i]["answer_index"] for i in ids])
mask = np.arange(10)[None] < nopt[:, None]
res = {}
for l in [12,16,18,20,21,22,23,24]:
    L = lens(np.load(W+f"h{l}.npy")); L[~mask] = -1e9
    pred = L.argmax(1); res[l] = L
    print(f"h{l}: acc {np.mean(pred==gold):.3f}  pred counts {np.bincount(pred,minlength=10)}")
L = res[24]; pred = L.argmax(1)
print("gold counts", np.bincount(gold, minlength=10))
np.save("work/p1_logits.npy", L)
# rank of gold among wrong
r = (L > L[np.arange(512), gold][:, None]).sum(1)
print("rank of gold (0=correct):", np.bincount(r, minlength=10))
# tokenization & option spans
offs = json.load(open(W+"meta.json"))["offsets_tok"] if "offsets_tok" in json.load(open(W+"meta.json")) else None
otok = np.load(W+"offsets_tok.npy") if offs is None else np.array(offs)
A = {l: np.load(W+f"attn_last_L{l}.npy") for l in [10,14,18,20,22]}
spans = []  # per question: list of (start_tok, end_tok, letter_tok)
for qi, i in enumerate(ids):
    s = render(i)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    om = enc["offset_mapping"]; n = len(om)
    assert n == otok[qi+1]-otok[qi], (n, otok[qi+1]-otok[qi])
    sp_i = []
    for k in range(nopt[qi]):
        c0 = s.index(f"\n({'ABCDEFGHIJ'[k]}) ")
        c1 = s.index("\n", c0+1)
        t0 = next(t for t in range(n) if om[t][1] > c0+1)  # first token covering '('
        tl = next(t for t in range(n) if om[t][1] > c0+2)  # letter token
        t1 = next(t for t in range(n) if om[t][0] >= c1)
        sp_i.append((t0, t1, tl))
    spans.append(sp_i)
json.dump(spans, open("work/p1_spans.json", "w"))
# attention mass per option for each layer/head
for l in [18, 20, 22]:
    M = np.zeros((512, 10, 16)); Ml = np.zeros((512, 10, 16))
    for qi in range(512):
        a = A[l][otok[qi]:otok[qi+1]]
        for k, (t0, t1, tl) in enumerate(spans[qi]):
            M[qi, k] = a[t0:t1].sum(0); Ml[qi, k] = a[tl]
    np.save(f"work/p1_optmass_L{l}.npy", M); np.save(f"work/p1_letmass_L{l}.npy", Ml)
    tot = M.sum(1)  # total attention on options
    for h in range(16):
        am = np.where(mask, M[:, :, h], -1).argmax(1)
        print(f"L{l}h{h:2d}: opt-mass {tot[:,h].mean():.2f} letter-tok {Ml[:,:,h].sum(1).mean():.2f} argmax==gold {np.mean(am==gold):.3f} argmax==pred {np.mean(am==pred):.3f}")
