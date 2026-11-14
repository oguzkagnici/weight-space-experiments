# H2 (pass 1 design): "knows but does not say". At forced-answer probes inside finished traces of the 15 MIXED questions (35% and 70% of the
# thinking, cut at a paragraph boundary), does gold lead at an intermediate layer / after the L22 router and get overturned by a later component
# (L22 MLP, L23 conv/MLP) in samples that end wrong? Records residual 12..24, w2 inputs 18..23, o_proj inputs 14..22, H12/L20 attention from the probe.
import json, numpy as np
from wsp.editor import *
SUF = "\n\nSo far, the most likely answer is ("
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
mixed = json.load(open("work/mixed_ids.json"))
FR = [0.35, 0.70]; MAXP = 3700
texts, index = [], []
for r in rows:
    if r["id"] not in mixed or r["cut"]: continue
    full = trace_text(r); pr = render_think(r["id"]); assert full.startswith(pr)
    k = full.find("</think>"); gen = full[len(pr):k if k > 0 else len(full)]
    for f in FR:
        c = int(len(gen) * f); b = gen.rfind("\n\n", 0, c)
        if b < len(gen) * 0.1: b = c
        pre = pr + gen[:b]
        ids = tok(pre, add_special_tokens=False).input_ids
        if len(ids) > MAXP:
            pre = tok.decode(ids[:MAXP]); b2 = pre.rfind("\n\n"); pre = pre[:b2] if b2 > len(pr) else pre
        texts.append(pre + SUF)
        index.append({"id": r["id"], "sample": r["sample"], "frac": f, "correct": r["correct"], "pred": r["pred"], "gold": r["gold"],
                      "cut_at_char": len(pre) - len(pr), "gen_chars": len(gen)})
ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
print("texts", len(texts), "tokens", sum(ntok), "max", max(ntok), "wrong samples", sum(1 for i in index if i["frac"] == 0.35 and not i["correct"]))
ATT = [14, 16, 18, 20, 22]
json.dump(index, open("work/probe1_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"hidden_states": list(range(12, 25)), "mlp": list(range(18, 24)), "inputs": [f"model.layers.{l}.self_attn.o_proj" for l in ATT],
                      "positions": "last", "topk": 20, "attn_last": [20, 22]}}, open("work/req_read1.json", "w"))
