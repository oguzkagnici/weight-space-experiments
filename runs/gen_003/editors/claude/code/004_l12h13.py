# hypothesis H7 (re-test + dose): scaling the output of L12.H13 (the main attention writer of the option-correctness
# belief, generation 2 ed_b) to FACTOR. Rebuilt from the published description: head 13 = input columns 13*128..14*128
# of model.layers.12.self_attn.o_proj.weight. usage: py code/004_l12h13.py FACTOR
import sys
from wsp.editor import *
f = float(sys.argv[1])
for n in edited(): drop_edit(n)
key = "model.layers.12.self_attn.o_proj.weight"; W = load_original(key)
W[:, 13*128:14*128] *= f
print(save_edit(key, W))
