# Hypothesis H1: at forced-answer probes in thinking context, some late-layer write (block or part of a w2 spectrum) lowers the gold letter; locate it by direct read-out, choose on split A, confirm on split B.
import json, numpy as np, torch, sys, glob, os
from wsp.editor import *
torch.set_num_threads(3)
D=sys.argv[1] if len(sys.argv)>1 else 'work/fwd_01'
meta=json.load(open(D+'/meta.json')); print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<200 else str(v)[:200]) for k,v in meta.items()})
print(sorted(os.listdir(D)))
