# Dose-response: the same gated neuron (L20 n5414) with a stronger write, target -15 instead of -9
# (w2 column x 15/9; gate and up rows unchanged). Prediction: more cut-with-statement traces close; risk: earlier closes
# at false-positive paragraph ends on finished traces.
import json, torch
from safetensors.torch import load_file
from wsp.editor import *
C = "/home/user/weight-space-experiments/runs/gen_005/editors/carry/candidates/gate_wrapup_L20/edited"
n = 5414
for t in ("w1", "w3"):
    name = f"model.layers.20.feed_forward.{t}.weight"; print(save_edit(name, load_file(f"{C}/{name}.safetensors")["tensor"].float()))
name = "model.layers.20.feed_forward.w2.weight"
W = load_file(f"{C}/{name}.safetensors")["tensor"].float(); O = load_original(name)
chg = (W != O).any(0).nonzero().flatten().tolist(); print("changed w2 columns:", chg)
W[:, n] = W[:, n] * (15 / 9)
print(save_edit(name, W))
json.dump({"type": "generate", "model": "both", "questions": json.load(open("work/dev32.json"))}, open("work/req_gen_d15.json", "w"))
