# Hypothesis: with H12/H15 queries projected off delta (code/015, alpha 1), the first-named advantage on the same 262 swap texts shrinks (real 0.85, syn 3.14 logits) while P(gold|pair) does not fall.
import json
r = json.load(open("work/req_p1.json"))
r["model"] = "edited"
r["return"] = {"positions": "last", "topk": 20, "hidden_states": [22, 24],
               "outputs": [f"model.layers.{l}.post_operator_norm" for l in range(24)] + [f"model.layers.{l}.post_ffn_norm" for l in range(24)],
               "inputs": ["model.layers.22.self_attn.o_proj", "model.layers.20.self_attn.o_proj"], "attn_last": [22]}
json.dump(r, open("work/req_p3.json", "w"))
