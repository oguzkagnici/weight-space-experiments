# Analyse pass 1 (H2): per-layer letter lens and late-component attribution at mixed-question probes; is gold preferred internally and
# pushed down later, more in samples that end wrong than in right ones?
import json, glob, numpy as np
from wsp.editor import *
D = "work/fwd_01/"; ix = json.load(open("work/probe1_index.json")); n = len(ix)
LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"
E = load("model.embed_tokens.weight")[LI].numpy(); Ec = E - E.mean(0, keepdims=True)
gf = load("model.embedding_norm.weight").numpy()
rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)
H = {l: np.load(D + f"h{l}.npy") for l in range(12, 25)}
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
def outp(t):
    p = np.zeros(10)
    for a, b in zip(tk[t], tl[t]):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    return p / max(p.sum(), 1e-9)
P = np.array([outp(t) for t in range(n)])
gold = np.array([LET.index(i["gold"]) for i in ix]); pick = np.array([LET.index(i["pred"]) if i["pred"] else -1 for i in ix])
corr = np.array([i["correct"] for i in ix]); frac = np.array([i["frac"] for i in ix])
lg24 = (H[24] / rms(H[24]) * gf) @ E.T
print("H24 lens argmax == output argmax:", np.mean(lg24.argmax(1) == P.argmax(1)).round(3))
print("output: probe argmax==gold  right-samples %.2f wrong-samples %.2f; ==final pick right %.2f wrong %.2f" % (
    np.mean(P.argmax(1)[corr == 1] == gold[corr == 1]), np.mean(P.argmax(1)[corr == 0] == gold[corr == 0]),
    np.mean(P.argmax(1)[corr == 1] == pick[corr == 1]), np.mean(P.argmax(1)[corr == 0] == pick[corr == 0])))
print("\nLENS by layer (own-RMS final norm): acc(argmax==gold) | P10(gold) mean, by sample outcome and probe point")
print("layer  " + "  ".join(f"{('R' if c else 'W')}{f:.2f}:acc  P(g)" for c in (1, 0) for f in (0.35, 0.70)))
for l in range(12, 25):
    lg = (H[l] / rms(H[l]) * gf) @ E.T; z = lg - lg.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = []
    for c in (1, 0):
        for f in (0.35, 0.70):
            m = (corr == c) & (frac == f)
            s.append(f"{np.mean(lg[m].argmax(1) == gold[m]):.2f}  {p[m, gold[m]].mean():.2f}")
    print(f"h{l:2d}   " + "     ".join(s))
# wrong samples: does gold lead (or rank above the eventual output argmax) at some layer >= 16 but not at the output?
print("\nWRONG-sample probes where output argmax != gold: layers at which lens argmax == gold")
m = (corr == 0) & (P.argmax(1) != gold)
for l in range(16, 25):
    lg = (H[l] / rms(H[l]) * gf) @ E.T
    print(f"  h{l}: gold top in {np.sum(lg[m].argmax(1) == gold[m])}/{m.sum()}; gold above output-top in {np.sum(lg[m, gold[m]] > lg[m, P[m].argmax(1)])}")
m2 = (corr == 1) & (P.argmax(1) == gold)
print("RIGHT-sample probes where output argmax == gold: share where some other letter tops the lens")
for l in range(16, 25):
    lg = (H[l] / rms(H[l]) * gf) @ E.T
    print(f"  h{l}: other top {np.sum(lg[m2].argmax(1) != gold[m2])}/{m2.sum()}")
# ---- component attribution (frozen final-norm scale), margin gold - output top (for probes where they differ) and centered gold
scale = gf / rms(H[24])
comp = {}
for l in range(18, 24):
    h = np.load(D + f"mlp{l}.npy"); W2 = load(f"model.layers.{l}.feed_forward.w2.weight").numpy(); gp = load(f"model.layers.{l}.post_ffn_norm.weight").numpy()
    y = h @ W2.T; comp[f"L{l} mlp"] = y / rms(y) * gp
for l in (14, 16, 18, 20, 22):
    f = glob.glob(D + f"*layers.{l}.self_attn.o_proj*")[0]; a = np.load(f)
    Wo = load(f"model.layers.{l}.self_attn.o_proj.weight").numpy(); gp = load(f"model.layers.{l}.post_operator_norm.weight").numpy()
    y = a @ Wo.T; ry = rms(y)
    for hd in range(16):
        sl = slice(128 * hd, 128 * hd + 128); comp[f"L{l} H{hd}"] = (a[:, sl] @ Wo[:, sl].T) * gp / ry
# check residual identity to locate indexing: h(l+1) - h(l) - mlp_l = mixer_l
for l in (20, 22):
    att = sum(comp[f"L{l} H{hd}"] for hd in range(16))
    for off in (0, 1):
        r = H[l + off + 1] - H[l + off] - comp[f"L{l} mlp"]
        print(f"identity check L{l} with h{l+off}->h{l+off+1}: rel err {np.linalg.norm(r - att) / np.linalg.norm(att):.3f}")
