# Hypothesis: L20 heads 9, 12, 13, 15 (attend more to the first-named finalist at a commit) carry the mention-order primacy; zeroing their o_proj input columns should shrink the swap effect (diagnostic edit only, not the candidate).
import torch
from wsp.editor import *
for n in list(edited()): drop_edit(n)
W=load_original('model.layers.20.self_attn.o_proj.weight').clone()
for h in (9,12,13,15): W[:,h*128:(h+1)*128]=0
save_edit('model.layers.20.self_attn.o_proj.weight',W); print(edited())
