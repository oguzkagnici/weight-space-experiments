# Pass 1 analysis (read, parent). For each in-think answer statement: where does the parent's L20 n5414 wrap-up gate
# stand at the tokens that end the statement line, and what does the parent want to write next?
# Groups: mid statements, split into "single" (one letter) vs "two" (", or H?" / "(C) or (D)"), and hedged vs firm;
# final_full = last statement in finished traces (the gate's own target).
# Also: a mean-difference direction for "single-letter mid statement just completed" vs other line ends, LOTO AUC.
import json, re, numpy as np, torch
from wsp.editor import *
W = "work/fwd_01"
M = json.load(open(f"{W}/meta.json")); P = json.load(open("work/p1_meta.json"))
texts = json.load(open("work/req_01.json"))["texts"]
X = np.load(f"{W}/in_model.layers.20.feed_forward.npy").astype(np.float32)
TI, TV = np.load(f"{W}/topk_ids.npy"), np.load(f"{W}/topk_logprobs.npy")
LP = np.load(f"{W}/token_logprobs.npy")
off = np.concatenate([[0], np.cumsum(M["n_tokens"])])
lpoff = np.concatenate([[0], np.cumsum(np.array(M["n_tokens"]) - 1)])
w1 = load_original("model.layers.20.feed_forward.w1.weight")[5414].numpy()
w3 = load_original("model.layers.20.feed_forward.w3.weight")[5414].numpy()
Z, U = X @ w1, X @ w3
A = Z / (1 + np.exp(-Z)) * U
print(f"all tokens: gate z pct50/99/99.8 {np.percentile(Z,50):+.2f}/{np.percentile(Z,99):+.2f}/{np.percentile(Z,99.8):+.2f}; "
      f"act>0.5 share {np.mean(A > 0.5):.4f}")
HEDGE = re.compile(r"\b(likely|maybe|perhaps|probably|might|guess|could|seems?|plausibl\w*|best)\b", re.I)
TWO = re.compile(r"^[^\n]{0,25}\bor\b\s*\(?[A-J]\b")
NL = {i for i in range(len(tok)) if "\n" in tok.decode([i])} if False else None
def is_nl(t): return "\n" in tok.decode([int(t)])
REOPEN = ["But", " But", "However", "Let", " Let", "Wait", " Wait", "Actually", "Alternatively", "Hmm", "Check", "Double", "Maybe", "Hold", "We", " We", "Now"]
CLOSE = ["Thus", "Hence", "Therefore", "So", "Final", "Answer", "</think>"]
def one(w):
    e = tok.encode(w, add_special_tokens=False); return e[0] if len(e) == 1 else None
RID = [x for x in map(one, REOPEN) if x is not None]; CID = [x for x in map(one, CLOSE) if x is not None]
THINK_END = one("</think>")
rows = []
for i, (p, t) in enumerate(zip(P, texts)):
    enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
    ids, om = enc["input_ids"], enc["offset_mapping"]
    assert len(ids) == M["n_tokens"][i]
    # token containing the last char of the statement ("... answer is (E)" ends at stmt_end)
    k_end = next(k for k, (a, b) in enumerate(om) if b >= p["stmt_end"])
    # first newline token at or after the statement end = decision point
    k_nl = next((k for k in range(k_end, len(ids)) if is_nl(ids[k])), None)
    line_start = t.rfind("\n", 0, p["stmt_char"]) + 1
    line = t[line_start:(om[k_nl][0] if k_nl is not None else p["stmt_end"] + 40)]
    tail = t[p["stmt_end"]:p["stmt_end"] + 30]
    two = bool(TWO.match(tail)) or bool(re.search(r"\bor\b\s*\(?[A-J]\b", t[p["stmt_end"]:om[k_nl][0]] if k_nl else tail))
    hedged = bool(HEDGE.search(line)) or "?" in t[p["stmt_end"]:(om[k_nl][0] if k_nl else p["stmt_end"] + 30)]
    g = off[i]
    win = range(k_end, (k_nl + 1) if k_nl is not None else k_end + 4)
    zmax = max(Z[g + k] for k in win); amax = max(A[g + k] for k in win)
    r = dict(p, two=two, hedged=hedged, line=line.strip()[-90:], k_nl=k_nl, zmax=float(zmax), amax=float(amax))
    if k_nl is not None:
        # next-token distribution AFTER the decision point = prediction at position k_nl
        ti, tv = TI[g + k_nl], np.exp(TV[g + k_nl])
        r["p_reopen"] = float(sum(v for a, v in zip(ti, tv) if a in RID)); r["p_close"] = float(sum(v for a, v in zip(ti, tv) if a in CID))
        r["p_think_end"] = float(sum(v for a, v in zip(ti, tv) if a == THINK_END))
        r["top"] = " | ".join(f"{tok.decode([int(a)])!r}:{v:.2f}" for a, v in zip(ti[:6], tv[:6]))
        r["actual_next"] = tok.decode([ids[k_nl + 1]]) if k_nl + 1 < len(ids) else None
        r["z_nl"], r["a_nl"] = float(Z[g + k_nl]), float(A[g + k_nl])
    rows.append(r)
def grp(name, sel):
    s = [r for r in rows if sel(r)]
    if not s: print(name, "none"); return
    zs = np.array([r.get("z_nl", np.nan) for r in s]); pr = np.array([r.get("p_reopen", np.nan) for r in s]); pc = np.array([r.get("p_close", np.nan) for r in s])
    print(f"{name:28s} n {len(s):2d} | gate z at newline med {np.nanmedian(zs):+.2f} (min {np.nanmin(zs):+.2f} max {np.nanmax(zs):+.2f}) | "
          f"amax med {np.median([r['amax'] for r in s]):+.2f} | P(reopen) {np.nanmean(pr):.2f} P(close) {np.nanmean(pc):.2f}")
print("== groups")
grp("final_full (firm, target)", lambda r: r["kind"] == "final_full")
grp("mid single-letter", lambda r: r["kind"] == "mid" and not r["two"])
grp("mid single hedged", lambda r: r["kind"] == "mid" and not r["two"] and r["hedged"])
grp("mid single unhedged", lambda r: r["kind"] == "mid" and not r["two"] and not r["hedged"])
grp("mid two-letter", lambda r: r["kind"] == "mid" and r["two"])
print("== mid statements")
for r in rows:
    if r["kind"] == "mid":
        print(f"{r['id']:14s} cut {r['cut']} two {int(r['two'])} hedged {int(r['hedged'])} z_nl {r.get('z_nl', float('nan')):+6.2f} a {r.get('a_nl', float('nan')):+5.2f} "
              f"Pre {r.get('p_reopen', 0):.2f} Pcl {r.get('p_close', 0):.2f} next {r.get('actual_next')!r} | ...{r['line'][-70:]!r}")
        print(f"{'':16s} top: {r.get('top')}")
# ---- direction: single-letter mid statement newline vs all other newline tokens (all texts), leave-one-text-out
pos, neg = [], []
for i, r in enumerate(rows):
    enc_ids = tok(texts[i], add_special_tokens=False)["input_ids"]; g = off[i]
    nls = [k for k in range(len(enc_ids)) if is_nl(enc_ids[k])]
    for k in nls:
        if r["kind"] == "mid" and not r["two"] and k == r["k_nl"]:
            pos.append((i, g + k))
        elif k != r["k_nl"]:
            neg.append((i, g + k))
print(f"== direction: {len(pos)} positives, {len(neg)} other newline tokens")
def auc(a, b):
    a, b = np.asarray(a), np.asarray(b); return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
Xn = X / np.sqrt((X ** 2).mean(1, keepdims=True))
sc_pos, sc_neg = [], []
for i in sorted({i for i, _ in pos}):
    tp = [k for j, k in pos if j != i]; tn = [k for j, k in neg if j != i]
    d = Xn[tp].mean(0) - Xn[tn].mean(0); d /= np.linalg.norm(d)
    sc_pos += [float(Xn[k] @ d) for j, k in pos if j == i]; sc_neg += [float(Xn[k] @ d) for j, k in neg if j == i]
print(f"LOTO AUC (held-out text's positive vs its own other newlines): {auc(sc_pos, sc_neg):.3f}")
d = Xn[[k for _, k in pos]].mean(0) - Xn[[k for _, k in neg]].mean(0); d /= np.linalg.norm(d)
allsc = Xn @ d
print(f"full-fit direction: pos med {np.median(allsc[[k for _, k in pos]]):+.2f}, other newlines med {np.median(allsc[[k for _, k in neg]]):+.2f}, "
      f"all tokens pct99/99.8 {np.percentile(allsc,99):+.2f}/{np.percentile(allsc,99.8):+.2f}")
print(f"cos(direction, n5414 gate row) {float(d @ w1 / np.linalg.norm(w1)):+.3f}")
# final-statement newlines vs the direction (they are excluded from positives; a unit on d would also fire there)
fin = [off[i] + r["k_nl"] for i, r in enumerate(rows) if r["kind"] == "final_full" and r["k_nl"] is not None]
print(f"final-statement newlines on d: med {np.median(allsc[fin]):+.2f}")
np.save("work/d_single_mid.npy", d)
json.dump(rows, open("work/p1_rows.json", "w"), indent=0)
