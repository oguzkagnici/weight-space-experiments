# DEV32 for the dose comparison of gate_wrapup (carry vs stronger): all 9 cut-with-statement traces (where the neuron acts),
# 15 cut-without-statement, 8 finished (harm check). Fixed, deterministic choice (sorted ids, every k-th).
import json
from wsp.editor import *
G = json.load(open("work/groups.json"))
dev = sorted(G["cut_stmt"]) + sorted(G["cut_nostmt"])[::3][:15] + sorted(G["finished"])[::5][:8]
assert len(dev) == 32 and len(set(dev)) == 32
json.dump(dev, open("work/dev32.json", "w"))
json.dump({"type": "generate", "model": "both", "questions": dev}, open("work/req_gen_carry.json", "w"))
print(dev)
