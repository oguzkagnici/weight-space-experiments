# Hypothesis: L12.H13 is the strongest attention writer of the option-end correctness belief (SNR .85); doubling its output (o_proj cols 1664:1792 x2) sharpens the belief and raises accuracy (causal test of the content path).
from wsp.editor import *
import sys, json
f=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
for n in edited(): drop_edit(n)
W=load_original('model.layers.12.self_attn.o_proj.weight').clone()
W[:,13*128:14*128]*=f
save_edit('model.layers.12.self_attn.o_proj.weight',W)
s=json.load(open('work/split.json'))
json.dump({"type":"score","model":"both","questions":s['dev'][1212:1468]},open('work/req_S1.json','w'))
print(edited())
