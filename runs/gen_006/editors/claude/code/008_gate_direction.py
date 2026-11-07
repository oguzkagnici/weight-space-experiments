# Analysis of pass 2 (hypothesis in code/007): is the "\n\n" after a two-finalist paragraph separable from other "\n\n"
# at the L21 MLP input? Mean-difference direction, leave-one-trace-out AUC; and the model's planned next tokens there.
import json, glob, os, re
import numpy as np
from wsp.editor import tok, render_think, trace_text
D = "work/fwd_02"
gm = json.load(open("work/gate_read_meta.json")); ids = gm["ids"]
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
meta1 = [m for m in json.load(open("work/probe1_meta.json")) if not m.get("control")]
print(sorted(os.listdir(D)))
X = np.load(glob.glob(f"{D}/in_*21*w1*.npy")[0], mmap_mode="r")
off = np.load(f"{D}/offsets.npy") if os.path.exists(f"{D}/offsets.npy") else np.load(glob.glob(f"{D}/offsets*.npy")[0])
topk_ids = np.load(f"{D}/topk_ids.npy", mmap_mode="r"); topk_lp = np.load(f"{D}/topk_logprobs.npy", mmap_mode="r")
print("X", X.shape, "offsets", off.shape, "topk", topk_ids.shape)
BREAK = {i for i in range(len(tok)) if "\n\n" in tok.decode([i])}   # tokens ending a paragraph (".\n\n", ":\n\n", ...)
print("break token types:", len(BREAK))
recs = []   # (text index, global row, label, frac, id)
for ti, q in enumerate(ids):
    r = rows[q]; text_prompt = render_think(q)
    raw = r["raw"]; body = raw[len("<think>"):] if raw.startswith("<think>") else raw
    think = body.split("</think>")[0]
    enc = tok(trace_text(r), return_offsets_mapping=True, add_special_tokens=False)
    n = off[ti + 1] - off[ti]
    if len(enc["input_ids"]) != n:
        print("token count mismatch", q, len(enc["input_ids"]), n); continue
    pos_chars = {len(text_prompt) + m["cut_at"] for m in meta1 if m["id"] == q}
    for j, (tid, (a, b)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
        if tid not in BREAK or a < len(text_prompt) or a >= len(text_prompt) + len(think): continue
        lab = 1 if any(a <= pc < b for pc in pos_chars) else 0
        recs.append((ti, off[ti] + j, lab, (a - len(text_prompt)) / len(think), q))
lab = np.array([x[2] for x in recs]); grp = np.array([x[0] for x in recs])
H = np.stack([np.asarray(X[x[1]], dtype=np.float32) for x in recs])
print(f"breaks: {len(recs)}, positives {lab.sum()} (expected {len(meta1)})")
def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)
scores = np.zeros(len(recs))
for g in np.unique(grp):
    tr = grp != g
    d = H[tr & (lab == 1)].mean(0) - H[tr & (lab == 0)].mean(0); d /= np.linalg.norm(d)
    scores[grp == g] = H[grp == g] @ d
print(f"leave-one-trace-out AUC (mean-diff direction): {auc(scores, lab):.3f}")
d = H[lab == 1].mean(0) - H[lab == 0].mean(0); d /= np.linalg.norm(d)
np.save("work/gate_dir_L21.npy", d)
s_all = H @ d
pos_s = np.sort(s_all[lab == 1]); neg_s = np.sort(s_all[lab == 0])
for q in (0.25, 0.5, 0.75):
    thr = np.quantile(pos_s, q)
    print(f" thr at pos q{int(q*100)} = {thr:.2f}: fires on {np.mean(pos_s >= thr):.2f} of positives, {np.mean(neg_s >= thr):.3f} of negatives "
          f"({int(np.sum(neg_s >= thr))} of {len(neg_s)}); cv-scores negatives above: {np.mean(scores[lab==0] >= np.quantile(scores[lab==1], q)):.3f}")
# negatives above threshold: in which kind of trace?
thr = np.quantile(pos_s, 0.25)
kinds = {"narrow": set(gm["narrow"]), "fin": set(gm["other_fin"]), "cut": set(gm["other_cut"])}
for k, s in kinds.items():
    m = np.array([x[4] in s for x in recs]) & (lab == 0)
    print(f"  negatives in {k} traces: {m.sum()}, above q25-threshold {int(np.sum(s_all[m] >= thr))}")
# what does the model plan to write after the break (next-token top-5 at positives)
print("\nplanned next tokens at positive breaks (top-5 with prob):")
for x in [x for x in recs if x[2] == 1][:20]:
    t5 = [(tok.decode([int(i)]).strip() or repr(tok.decode([int(i)])), round(float(np.exp(l)), 2)) for i, l in zip(topk_ids[x[1]][:5], topk_lp[x[1]][:5])]
    print(f"  {x[4]:14s} @{x[3]:.2f} {t5}")
json.dump([{"ti": int(a), "row": int(b), "lab": int(c), "frac": float(f), "id": q} for a, b, c, f, q in recs], open("work/breaks.json", "w"))
np.save("work/break_scores.npy", s_all)
