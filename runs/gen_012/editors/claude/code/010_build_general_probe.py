# H2b + H9 (pass 2 design, original model): (a) generalise H2 beyond mixed questions: forced-answer probes at 50% and at the END of the
# thinking (just before </think>) in finished WRONG traces of all other passk/bank questions (+ right controls at 50%), residual 14..24 -> is
# gold ever preferred internally when the conclusion is wrong? (b) pre-thinking prior probes on 240 practice questions (not bank/holdout)
# -> letter-preference statistics of the prior readout, for a cross-fitted per-letter correction (H9: the early latent belief anchors on
# this prior; a less biased prior should raise P(gold) at prior probes).
import json, random, numpy as np
from wsp.editor import *
SUF = "\n\nSo far, the most likely answer is ("; SUF0 = "So far, the most likely answer is ("
mixed = set(json.load(open("work/mixed_ids.json")))
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [dict(json.loads(l), sample=0) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
seen = set(); R = []
for r in rows + bank:
    k = (r["id"], r["sample"])
    if k in seen or r["cut"] or r["id"] in mixed: continue
    seen.add(k); R.append(r)
def probe(r, f):
    full = trace_text(r); pr = render_think(r["id"]); k = full.find("</think>"); gen = full[len(pr):k if k > 0 else len(full)]
    if f >= 1.0: pre = pr + gen.rstrip()
    else:
        c = int(len(gen) * f); b = gen.rfind("\n\n", 0, c); pre = pr + gen[:b if b > 0 else c]
    n = len(tok(pre, add_special_tokens=False).input_ids)
    return pre + SUF, n
random.seed(12)
wrong = [r for r in R if not r["correct"]]; right = [r for r in R if r["correct"]]; random.shuffle(right)
texts, index, budget = [], [], 100000
for r in sorted(wrong, key=lambda r: r["tokens"]):
    for f in (0.5, 1.0):
        t, n = probe(r, f)
        if n > 3990 or budget - n < 0: continue
        texts.append(t); index.append({"kind": "gen", "id": r["id"], "sample": r["sample"], "frac": f, "correct": 0, "pred": r["pred"], "gold": r["gold"]}); budget -= n
nw = len(texts); budget = 30000
for r in sorted(right[:60], key=lambda r: r["tokens"]):
    t, n = probe(r, 0.5)
    if n > 3990 or budget - n < 0: continue
    texts.append(t); index.append({"kind": "gen", "id": r["id"], "sample": r["sample"], "frac": 0.5, "correct": 1, "pred": r["pred"], "gold": r["gold"]}); budget -= n
excl = set(json.load(open("/home/user/weight-space-experiments/runs/shared/holdout48_ids.json"))) | set(json.load(open("/home/user/weight-space-experiments/runs/shared/holdout_bank_ids.json")))
excl |= {r["id"] for r in rows + bank}
Q = questions(); pool = sorted(q for q in Q if q not in excl); random.shuffle(pool)
tot = 199000 - sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts); npri = 0
for q in pool:
    t = render_think(q) + SUF0; n = len(tok(t, add_special_tokens=False).input_ids)
    if tot - n < 0 or npri >= 400: break
    texts.append(t); index.append({"kind": "prior", "id": q, "gold": Q[q]["answer"]}); tot -= n; npri += 1
print("wrong-trace probes", nw, "right probes", sum(1 for i in index if i["kind"] == "gen") - nw, "prior probes", npri, "tokens left", tot,
      "| wrong traces", len({(i['id'], i['sample']) for i in index if i['kind'] == 'gen' and i['correct'] == 0}), "questions", len({i['id'] for i in index if i['kind'] == 'gen' and i['correct'] == 0}))
json.dump(index, open("work/probe2_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"hidden_states": [14, 15, 16, 18, 20, 22, 23, 24], "positions": "last", "topk": 20}}, open("work/req_read2.json", "w"))
