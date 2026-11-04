# Hypothesis: none (build read request 1: parent, DEV[:160], residual at all positions for probing option-level correctness).
import json
s=json.load(open('work/split.json'))
req={"type":"read","model":"original","questions":s['dev'][:160],
 "return":{"hidden_states":[14,18,20,22,24],"outputs":["model.layers.22.self_attn.k_norm"],
 "positions":"all","attn_last":[2,4,6,8,10,12,14,16,18,20,22]}}
json.dump(req,open('work/req1.json','w'))
