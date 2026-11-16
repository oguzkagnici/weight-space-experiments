# Which L23 MLP neurons did the parent's accepted edits change (to avoid reusing them)? Are 881/331 still near-duplicates?
import torch
from wsp.editor import *
for nm in ("w1", "w3"):
    a = load_original(f"model.layers.23.feed_forward.{nm}.weight"); b = load_base(f"model.layers.23.feed_forward.{nm}.weight")
    print(nm, "changed rows", torch.nonzero((a - b).abs().sum(1) > 0).ravel().tolist()[:20])
a = load_original("model.layers.23.feed_forward.w2.weight"); b = load_base("model.layers.23.feed_forward.w2.weight")
print("w2 changed cols", torch.nonzero((a - b).abs().sum(0) > 0).ravel().tolist()[:20])
w1 = load_original("model.layers.23.feed_forward.w1.weight"); w3 = load_original("model.layers.23.feed_forward.w3.weight")
cos = lambda x, y: float(x @ y / x.norm() / y.norm())
print("cos 881/331 w1", round(cos(w1[881], w1[331]), 5), "w3", round(cos(w3[881], w3[331]), 5))
