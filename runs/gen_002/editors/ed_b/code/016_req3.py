# Hypothesis: none (build score request 3: parent on DEV[700:1212] with last-position activations for letter-stage decomposition).
import json
s=json.load(open('work/split.json'))
req={"type":"score","model":"original","questions":s['dev'][700:1212],
 "activations":{"hidden_states":[18,20,21,22,23,24],"mlp":[20,21,22,23],"inputs":["model.layers.22.self_attn.o_proj"]}}
json.dump(req,open('work/req3.json','w'))
