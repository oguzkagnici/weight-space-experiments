# Hypothesis: at the L22/L23 MLP input, a mean-difference direction separates paragraph-break tokens that end a
# two-finalist paragraph from all other paragraph breaks (gen 6 at L21: leave-one-trace-out AUC 0.86), and it stays far
# below threshold on ordinary tokens, so a sharp gate can be nearly silent off-target.
import json, re, sys
import numpy as np
from wsp.editor import *
from sklearn.metrics import roc_auc_score
X = r"\(?([A-J])\)?"
pair = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + X + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + X + r"\b"
                  r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + X + r"\s+or\s+" + X + r"\b"
                  r"|\b" + X + r"\s+(?:or|vs\.?)\s+" + X + r"\s*[\?\.]", re.I)
rows = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl")}
dev = json.load(open("work/dev_ids.json"))["dev"]
off = np.load("work/fwd_02/offsets.npy")
breaks = []          # (global row, trace index, label, frac)
for ti, qid in enumerate(dev):
    text = trace_text(rows[qid]); p = render_think(qid)
    enc = tok.encode(text, add_special_tokens=False)
    n_rows = int(off[ti + 1] - off[ti]); shift = n_rows - len(enc.ids if hasattr(enc, "ids") else enc)
    ids = enc.ids if hasattr(enc, "ids") else enc
    # character offsets of each token
    spans = enc.offsets if hasattr(enc, "offsets") else None
    if spans is None:
        spans = []; pos = 0
        for k in range(len(ids)):
            s = tok.decode(ids[:k + 1]); spans.append((pos, len(s))); pos = len(s)
    think_end = text.find("</think>", len(p)); think_end = len(text) if think_end < 0 else think_end
    think = text[len(p):think_end]
    pos_ends = set()
    for m in pair.finditer(think):
        if len({x for x in m.groups() if x}) == 2:
            e = think.find("\n\n", m.end())
            if e >= 0: pos_ends.add(len(p) + e)
    for k, (a, b) in enumerate(spans):
        if a < len(p) or a >= think_end: continue
        piece = text[a:b]
        if "\n\n" in piece:
            j = text.find("\n\n", a)
            lab = int(j in pos_ends)
            breaks.append((int(off[ti]) + k + shift, ti, lab, (a - len(p)) / max(1, think_end - len(p))))
B = np.array(breaks)
print(f"shift {shift}; breaks {len(B)}, positives {int(B[:, 2].sum())} in {len(set(B[B[:, 2] == 1][:, 1]))} traces")
json.dump([dict(row=int(r), ti=int(t), lab=int(l), frac=float(f)) for r, t, l, f in breaks], open("work/breaks.json", "w"))
for L in (22, 23):
    Xa = np.load(f"work/fwd_02/model.layers.{L}.feed_forward.in.npy") if False else None
    import glob
    f = [x for x in glob.glob("work/fwd_02/*.npy") if f"layers.{L}.feed_forward" in x][0]
    Xa = np.load(f, mmap_mode="r")
    rr = B[:, 0].astype(int); lab = B[:, 2].astype(int); tr = B[:, 1].astype(int)
    xb = np.asarray(Xa[rr], dtype=np.float32)
    scores = np.zeros(len(rr))
    for t in np.unique(tr):                                 # leave-one-trace-out direction
        m = tr != t
        d = xb[m & (lab == 1)].mean(0) - xb[m & (lab == 0)].mean(0); d /= np.linalg.norm(d)
        scores[tr == t] = xb[tr == t] @ d
    d = xb[lab == 1].mean(0) - xb[lab == 0].mean(0); d /= np.linalg.norm(d)
    np.save(f"work/gate_dir_L{L}.npy", d)
    # all-token distribution (subsample) and bias proxy
    idx = np.arange(0, Xa.shape[0], 7)
    xa = np.asarray(Xa[idx], dtype=np.float32)
    mu = xa.mean(0); mu_hat = mu / np.linalg.norm(mu); mp = float((xa @ mu_hat).mean()); cv = float((xa @ mu_hat).std() / mp)
    np.save(f"work/mu_hat_L{L}.npy", mu_hat)
    sa = xa @ d; sp = xb[lab == 1] @ d; sn = xb[lab == 0] @ d
    print(f"L{L} ({f.split('/')[-1]}): LOTO AUC {roc_auc_score(lab, scores):.3f}; full-fit d.x: positives median {np.median(sp):.2f} "
          f"(p25 {np.percentile(sp, 25):.2f}), other breaks p99 {np.percentile(sn, 99):.2f}, all tokens p99 {np.percentile(sa, 99):.2f} "
          f"p99.9 {np.percentile(sa, 99.9):.2f} max {sa.max():.2f}; mu proj mean {mp:.2f} (cv {cv:.3f})")
    json.dump({"mp": mp, "cv": cv}, open(f"work/mu_L{L}.json", "w"))
