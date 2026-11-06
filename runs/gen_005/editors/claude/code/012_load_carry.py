# Set my edit to the carried gen-4 candidate gate_wrapup_L20 (exact copy), to measure it on the 96 det1 trace-bank questions
# as the paired baseline for dose variants. (Its hidden selection result in gen 5: +9.4 points, z 3.6, +12/-0.)
import json
from safetensors.torch import load_file
from wsp.editor import *
C = "/home/user/weight-space-experiments/runs/gen_005/editors/carry/candidates/gate_wrapup_L20/edited"
for n in ("w1", "w2", "w3"):
    name = f"model.layers.20.feed_forward.{n}.weight"
    print(save_edit(name, load_file(f"{C}/{name}.safetensors")["tensor"].float()))
B = [json.loads(l)["id"] for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")]
json.dump({"type": "generate", "model": "both", "questions": B}, open("work/req_gen_carry.json", "w"))
print(edited(), len(B))
