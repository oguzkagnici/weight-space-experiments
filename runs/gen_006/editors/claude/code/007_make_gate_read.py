# Hypothesis: the paragraph break ("\n\n", token 8) right after a two-finalist statement has a residual state that is
# linearly separable (at the L21 MLP input) from other paragraph breaks, so a gated neuron can fire there only.
# Builds pass 2: read the parent along the 22 narrowing traces + 10 other bank traces (5 finished, 5 cut), recording the
# L21 MLP input (w1 input, after ffn_norm) at all positions and the top-20 next tokens.
import json, random
from wsp.editor import trace_text
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
meta1 = json.load(open("work/probe1_meta.json"))
narrow = sorted({m["id"] for m in meta1 if not m.get("control")})
rest = [q for q in sorted(rows) if q not in narrow]
rng = random.Random("claude-g6-gate")
fin = rng.sample([q for q in rest if not rows[q]["cut"]], 5); cut = rng.sample([q for q in rest if rows[q]["cut"]], 5)
ids = narrow + fin + cut
texts = [trace_text(rows[q]) for q in ids]
json.dump({"ids": ids, "narrow": narrow, "other_fin": fin, "other_cut": cut}, open("work/gate_read_meta.json", "w"))
req = {"type": "read", "model": "original", "texts": texts,
       "return": {"inputs": ["model.layers.21.feed_forward.w1"], "positions": "all", "topk": 20, "token_logprobs": True}}
json.dump(req, open("work/req_02.json", "w"))
print(len(ids), "texts")
