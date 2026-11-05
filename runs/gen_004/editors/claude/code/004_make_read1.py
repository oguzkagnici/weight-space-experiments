# Pass 1 request: parent, last position of each decision-point text: residual stream, MLP neurons, per-head o_proj inputs, top-k, attention from the last token.
import json, random
sel = json.load(open("work/decision_points.json")); texts = json.load(open("work/decision_texts.json"))
keep = [i for i, p in enumerate(sel) if p["kind"] == "commit"]
rnd = random.Random(1)
for lab in ("reopen", "close"):
    idx = [i for i, p in enumerate(sel) if p["kind"] == "control" and p["label"] == lab]
    keep += sorted(rnd.sample(idx, 25))
sel2 = [sel[i] for i in keep]; texts2 = [texts[i] for i in keep]
json.dump(sel2, open("work/read1_points.json", "w"), indent=1)
attn = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
req = {"type": "read", "model": "original", "texts": texts2,
       "return": {"topk": 30, "positions": "last", "hidden_states": list(range(25)), "mlp": list(range(24)),
                  "inputs": [f"model.layers.{l}.self_attn.o_proj" for l in attn], "attn_last": [10, 14, 18, 22]}}
json.dump(req, open("work/req_read1.json", "w"))
print(len(texts2), "texts")
