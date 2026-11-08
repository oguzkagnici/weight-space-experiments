# Hypothesis: the token that ENDS a single-letter answer statement inside the thinking ("... answer is (E)." / "is J?")
# is separable at the L20 MLP input from every other token (incl. option-sweep letters, two-letter statements
# "B or H?", the prompt's "The answer is (X)" instruction and the visible answer after </think>), so a gated L20 unit
# can push </think> exactly there. Also: what does the parent predict there now (P(</think>), top tokens)?
import json, re, numpy as np
from wsp.editor import *
W = "work/fwd_01"
M = json.load(open(f"{W}/meta.json")); P = json.load(open("work/p1_meta.json"))
texts = json.load(open("work/req_01.json"))["texts"]
X = np.load(f"{W}/in_model.layers.20.feed_forward.npy").astype(np.float32)
TI, TV = np.load(f"{W}/topk_ids.npy"), np.load(f"{W}/topk_logprobs.npy")
off = np.concatenate([[0], np.cumsum(M["n_tokens"])])
TE = tok.encode("</think>", add_special_tokens=False); assert len(TE) == 1; TE = TE[0]
ANS = re.compile(r"answer is:?\s*\(?\*?\*?([A-J])\b\)?")
TWO = re.compile(r"^[\)\*\s,]*(or|and|/)\s*\(?[A-J]\b")
PUN = re.compile(r"^[\)\.\?\!\*\"\:]+$")
lab = np.zeros(len(X), np.int8)        # 1 single-letter stmt end in think, 2 two-letter stmt letter, 3 prompt, 4 after </think>
info = []
for i, t in enumerate(texts):
    enc = tok(t, add_special_tokens=False, return_offsets_mapping=True); om = enc["offset_mapping"]; ids = enc["input_ids"]
    g = off[i]; ps = len(render_think(P[i]["id"])) - len("<think>")      # think region starts here
    te = t.find("</think>", ps); te = len(t) if te < 0 else te
    for k, (a, b) in enumerate(om):
        if b <= ps: lab[g + k] = 3
        elif a >= te: lab[g + k] = 4
    for m in ANS.finditer(t, ps, te):
        k = next(k for k, (a, b) in enumerate(om) if b >= m.end())         # token holding the statement's last char
        while k + 1 < len(ids) and PUN.match(tok.decode([ids[k + 1]]).strip() or "x") and om[k + 1][0] < te:
            k += 1                                                          # extend over trailing ")." "?" tokens
        if TWO.match(t[m.end():m.end() + 20]):
            lab[g + k] = 2; continue
        lab[g + k] = 1
        nxt = tok.decode([ids[k + 1]]) if k + 1 < len(ids) else None
        info.append({"text": i, "id": P[i]["id"], "kind": P[i]["kind"], "pos": int(g + k), "frac": m.start() / te,
                     "ctx": t[max(m.start() - 30, 0):om[k][1]], "next": nxt,
                     "p_end": float(sum(np.exp(v) for a, v in zip(TI[g + k], TV[g + k]) if a == TE)),
                     "top3": [(tok.decode([int(a)]), round(float(np.exp(v)), 2)) for a, v in zip(TI[g + k][:3], TV[g + k][:3])]})
pos = np.where(lab == 1)[0]; print(f"{len(pos)} single-letter statement ends, {np.sum(lab == 2)} two-letter, {np.sum(lab == 3)} prompt tok, {np.sum(lab == 4)} post-think tok")
fin = [r for r in info if r["next"] and "</think>" in r["next"]]
print(f"parent P(</think>) at statement ends: next is </think> in {len(fin)}: median {np.median([r['p_end'] for r in fin]):.2f}; "
      f"others ({len(info) - len(fin)}): median {np.median([r['p_end'] for r in info if r not in fin]):.3f}")
for r in info:
    if r not in fin:
        print(f"  {r['id']:14s} {r['kind'][:3]} frac {r['frac']:.2f} P(end) {r['p_end']:.3f} next {r['next']!r:12s} ctx {r['ctx'][-50:]!r}")
# ---- direction: leave-one-text-out mean difference on RMS-normalised inputs
Xn = X / np.sqrt((X ** 2).mean(1, keepdims=True))
neg = np.where(lab != 1)[0]
txt = np.repeat(np.arange(len(texts)), M["n_tokens"])
def auc(a, b):
    a = np.sort(np.asarray(a)); b = np.asarray(b); return float(np.searchsorted(a, b, side="left").sum() / (len(a) * len(b)))  # P(b < a)
sp, sn = [], []
mu_neg = Xn[neg].mean(0)
for i in sorted(set(txt[pos])):
    tp = pos[txt[pos] != i]
    d = Xn[tp].mean(0) - mu_neg; d /= np.linalg.norm(d)
    sp += list(Xn[pos[txt[pos] == i]] @ d); sn += list(Xn[(txt == i) & (lab != 1)] @ d)
sp, sn = np.array(sp), np.array(sn)
print(f"LOTO: AUC {auc(sp, sn):.4f}; positives min/med {sp.min():+.1f}/{np.median(sp):+.1f}; negatives (same texts) 99.9/99.99/max "
      f"{np.percentile(sn, 99.9):+.1f}/{np.percentile(sn, 99.99):+.1f}/{sn.max():+.1f}")
for q in (5, 10, 25):
    th = np.percentile(sp, q); print(f"  threshold at positive pct {q} ({th:+.1f}): negative tokens above {np.sum(sn > th)} of {len(sn)}")
d = Xn[pos].mean(0) - mu_neg; d /= np.linalg.norm(d); s = Xn @ d
print(f"full fit: positives min/med {s[pos].min():+.1f}/{np.median(s[pos]):+.1f}; by label max: "
      + ", ".join(f"{n} {s[lab == l].max():+.1f}" for l, n in ((0, 'other-think'), (2, 'two-letter'), (3, 'prompt'), (4, 'post-think'))))
top = np.argsort(-np.where(lab != 1, s, -1e9))[:12]
for k in top:
    i = txt[k]; kk = k - off[i]; ids = tok(texts[i], add_special_tokens=False)["input_ids"]
    print(f"  top negative {s[k]:+.1f} lab {lab[k]} {P[i]['id']}: {tok.decode(ids[max(kk - 10, 0):kk + 1])[-60:]!r}")
np.save("work/d_stmt_end.npy", d); np.save("work/lab_p1.npy", lab); json.dump(info, open("work/stmt_info.json", "w"))
