# H14 (pass 2 design): samples of the same mixed question split EARLY (first ~100-400 generated tokens), at a content token that the model itself rated lower than an alternative (first recall/hypothesis).
# Probes "\n\nSo far, the most likely answer is (" after 25/50/100/200/400/800 generated tokens of each finished mixed-question sample, + one probe right after "<think>" per question.
import json, numpy as np
from wsp.editor import *
SUF = "\n\nSo far, the most likely answer is ("
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
mixed = json.load(open("work/mixed_ids.json"))
CK = [25, 50, 100, 200, 400, 800]
texts, index = [], []
for q in mixed:
    pr = render_think(q); texts.append(pr + SUF.lstrip("\n")); index.append({"id": q, "sample": None, "ck": 0})
for r in rows:
    if r["id"] not in mixed or r["cut"]: continue
    full = trace_text(r); pr = render_think(r["id"]); assert full.startswith(pr)
    pids = tok(pr, add_special_tokens=False).input_ids; gids = tok(full, add_special_tokens=False).input_ids
    assert gids[:len(pids)] == pids
    for c in CK:
        pre = tok.decode(gids[:len(pids) + c])
        texts.append(pre + SUF); index.append({"id": r["id"], "sample": r["sample"], "ck": c, "correct": r["correct"], "pred": r["pred"], "gold": r["gold"]})
ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
print("texts", len(texts), "tokens", sum(ntok), "max", max(ntok), "samples", len({(i['id'], i['sample']) for i in index if i['sample'] is not None}))
json.dump(index, open("work/probe2_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"token_logprobs": True, "topk": 10, "positions": "all"}}, open("work/req_read2.json", "w"))
