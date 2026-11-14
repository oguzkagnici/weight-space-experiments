# H3: in thinking mode the forced answer is readable from h15 on (after L14), i.e. the trace's STATED belief is copied early. Which heads write
# the selected letter (direct logit attribution, centered over letters, frozen final-norm scale), and do any components push toward gold in
# wrong samples (latent support) that the output does not show?
import json, glob, numpy as np
from wsp.editor import *
D = "work/fwd_01/"; ix = json.load(open("work/probe1_index.json")); n = len(ix)
LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"
E = load("model.embed_tokens.weight")[LI].numpy(); Ec = E - E.mean(0, keepdims=True)
gf = load("model.embedding_norm.weight").numpy(); rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)
H24 = np.load(D + "h24.npy"); scale = gf / rms(H24)
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
def outp(t):
    p = np.zeros(10)
    for a, b in zip(tk[t], tl[t]):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    return p / max(p.sum(), 1e-9)
P = np.array([outp(t) for t in range(n)]); top = P.argmax(1)
gold = np.array([LET.index(i["gold"]) for i in ix]); corr = np.array([i["correct"] for i in ix])
comp = {}
for l in (14, 16, 18, 20, 22):
    a = np.load(glob.glob(D + f"*layers.{l}.self_attn.o_proj*")[0])
    Wo = load(f"model.layers.{l}.self_attn.o_proj.weight").numpy(); gp = load(f"model.layers.{l}.post_operator_norm.weight").numpy()
    ry = rms(a @ Wo.T)
    for hd in range(16):
        sl = slice(128 * hd, 128 * hd + 128); comp[f"L{l}H{hd}"] = (a[:, sl] @ Wo[:, sl].T) * gp / ry
for l in range(18, 24):
    h = np.load(D + f"mlp{l}.npy"); W2 = load(f"model.layers.{l}.feed_forward.w2.weight").numpy(); gp = load(f"model.layers.{l}.post_ffn_norm.weight").numpy()
    y = h @ W2.T; comp[f"L{l}mlp"] = y / rms(y) * gp
# total per layer from residual diffs (includes conv layers)
H = {l: np.load(D + f"h{l}.npy") for l in range(12, 25)}
for l in range(12, 24): comp[f"layer{l}"] = H[l + 1] - H[l]
def dla(v, tgt): return ((v * scale) @ Ec.T)[np.arange(n), tgt]          # centered letter logit to target letter
rows = []
for k, v in comp.items():
    a_top = dla(v, top); a_gold = dla(v, gold)
    w = (corr == 0) & (top != gold)
    rows.append((k, a_top.mean(), a_gold[w].mean(), (a_gold[w] - a_top[w]).mean()))
print("component      to-selected(all)   wrong&top!=gold: to-gold   gold-minus-top")
for k, a, b, c in sorted(rows, key=lambda x: -abs(x[1]))[:25]: print(f"  {k:10s} {a:+7.2f}   {b:+7.2f}   {c:+7.2f}")
print("\nlargest positive gold-minus-top (latent support for gold in wrong probes):")
for k, a, b, c in sorted(rows, key=lambda x: -x[3])[:10]: print(f"  {k:10s} {c:+7.2f}  (to selected {a:+.2f})")
