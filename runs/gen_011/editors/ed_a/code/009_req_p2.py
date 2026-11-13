# Hypothesis: the +1.2-logit first-named advantage written by L22.H12 comes from specific source positions (the pair letters vs prompt option lines); recording L22 values and keys at all positions on the same 262 texts lets us decompose H12's output by source and estimate the key direction that marks the first-named letter.
import json
r = json.load(open("work/req_p1.json"))
r["return"] = {"positions": "all", "outputs": ["model.layers.22.self_attn.v_proj", "model.layers.22.self_attn.k_norm"]}
json.dump(r, open("work/req_p2.json", "w"))
print(len(r["texts"]))
