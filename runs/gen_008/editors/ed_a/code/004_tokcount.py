# Utility: token counts of bank trace_texts (to plan the read pass budget of 200k tokens).
import json
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
n=[len(tok.encode(trace_text(b)).ids) if hasattr(tok.encode(trace_text(b)),'ids') else len(tok.encode(trace_text(b))) for b in B]
import numpy as np
print(sum(n), np.mean(n), max(n), sum(x for x,b in zip(n,B) if b['cut']), sum(x for x,b in zip(n,B) if not b['cut']))
t=trace_text(B[0]); print(repr(t[:400])); print(repr(t[-200:]))
print(type(tok))
