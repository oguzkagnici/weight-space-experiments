# hypothesis H5 (edit): give the weak, partly independent "early routers" (L20.H7, L14.H15, L14.H12) more say by scaling
# their output (their 128 o_proj input columns) by FACTOR. usage: py code/003_amp_heads.py FACTOR [L22H15_FACTOR]
import sys
from wsp.editor import *
f = float(sys.argv[1]); f15 = float(sys.argv[2]) if len(sys.argv) > 2 else None
heads = {20: [7], 14: [15, 12]}
if f15: heads.setdefault(22, []).append(15)
for name in edited(): drop_edit(name)
for l, hs in heads.items():
    key = f"model.layers.{l}.self_attn.o_proj.weight"; W = load_original(key)
    for h in hs:
        W[:, h*128:(h+1)*128] *= (f15 if (l == 22 and h == 15) else f)
    print(save_edit(key, W))
