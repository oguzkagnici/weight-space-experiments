# Hypothesis: L22.H10 (the only head whose right>wrong stem attention at judging positions replicates on both halves) carries stem information
# into the decision; amplifying its output (o_proj input columns 1280:1408 x1.5) should raise P(gold) where the cue matters. Expected ~null (gate).
import sys, torch
from wsp.editor import *
name = "model.layers.22.self_attn.o_proj.weight"
W = load_original(name); print(name, tuple(W.shape))
norms = W.view(W.shape[0], 16, 128).norm(dim=(0, 2)); print("per-head column norms:", [round(float(x), 2) for x in norms])
s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
W2 = W.clone(); W2[:, 10 * 128:11 * 128] *= s
save_edit(name, W2); print("edited:", edited())
