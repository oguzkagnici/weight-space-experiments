# Hypothesis: the residual stream carries a linearly readable "how far into the trace" (length) signal.
# If so, a gated neuron can trigger a wrap-up only late in a trace (>~2,200 tokens), where finished traces
# are only ~33% correct anyway and cut traces score 0, while the forced-answer probe (pass 1) gives 26-44%.
import json
from wsp.editor import *
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
G = json.load(open("work/groups.json"))
cut = G["cut_nostmt"][:7]
fin = sorted([q for q in G["finished"]], key=lambda q: -B[q]["tokens"])[:7]   # longest finished traces
sel = cut + fin
texts = [trace_text(B[q]) for q in sel]
n = [len(tok(x, add_special_tokens=False).input_ids) for x in texts]
print(sel, n, sum(n))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"hidden_states": [14, 20], "positions": "all"}}, open("work/req_pos.json", "w"))
json.dump({"ids": sel, "n": n, "prompt_len": [len(tok(render_think(q), add_special_tokens=False).input_ids) for q in sel]}, open("work/pos_meta.json", "w"))
