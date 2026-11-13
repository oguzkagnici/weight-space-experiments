# H11 (pass 1 design): an exam-meta paragraph ("multiple choice / answer key / the question expects") moves the model's latent answer AWAY from gold more than an ordinary paragraph does.
# Probes (forced answer "So far, the most likely answer is (") at: p0 = before the preceding paragraph, p1 = before the meta paragraph, p2 = after it.
import json, re, numpy as np, random
from wsp.editor import *
META = re.compile(r"(multiple[- ]choice|in such questions|typical (exam|test|mcq)|answer key|test[- ]maker|(question|they|exam|test|problem) (likely|probably|might|may)? ?(expects?|wants?|intends?|is looking for))", re.I)
SUF = "\n\nSo far, the most likely answer is ("
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pk = {(r["id"], r["sample"]) for r in rows}; rows += [r for r in bank if (r["id"], 0) not in pk]
cands = []
for r in rows:
    if r["cut"]: continue
    full = trace_text(r); pr = render_think(r["id"]); assert full.startswith(pr)
    think = full[len(pr):]
    m = META.search(think)
    if not m: continue
    s = think.rfind("\n\n", 0, m.start()); s = 0 if s < 0 else s       # start of meta paragraph
    e = think.find("\n\n", m.end()); e = len(think) if e < 0 else e      # end of meta paragraph
    s0 = think.rfind("\n\n", 0, max(0, s - 1)); s0 = 0 if s0 < 0 else s0   # start of preceding paragraph
    if s0 >= s or s == 0: continue
    pts = [s0, s, e]
    texts = [pr + think[:p].rstrip() + SUF for p in pts]
    ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    if max(ntok) > 4090: continue
    cands.append({"id": r["id"], "sample": r.get("sample", 0), "correct": r["correct"], "pred": r["pred"], "gold": r["gold"],
                  "meta_frac": m.start() / len(think), "para": think[s:e][:200], "ntok": ntok, "texts": texts})
print("candidate traces", len(cands), "right", sum(c["correct"] for c in cands), "wrong", sum(1 - c["correct"] for c in cands))
# budget 200k tokens: all wrong first, then right, shortest first within each
random.seed(0)
W = sorted([c for c in cands if not c["correct"]], key=lambda c: sum(c["ntok"])); R = sorted([c for c in cands if c["correct"]], key=lambda c: sum(c["ntok"]))
order = [x for pair in zip(W, R) for x in pair] + W[len(R):] + R[len(W):]
sel, tot = [], 0
for c in order:
    if tot + sum(c["ntok"]) > 198000: continue
    sel.append(c); tot += sum(c["ntok"])
print("selected", len(sel), "right", sum(c["correct"] for c in sel), "wrong", sum(1 - c["correct"] for c in sel), "tokens", tot)
texts = [t for c in sel for t in c["texts"]]
json.dump([{k: v for k, v in c.items() if k != "texts"} for c in sel], open("work/probe1_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"topk": 50, "positions": "last"}}, open("work/req_read1.json", "w"))
print("texts", len(texts))
