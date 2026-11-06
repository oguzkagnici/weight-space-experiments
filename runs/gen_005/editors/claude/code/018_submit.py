"""Submit the stronger-dose variant of the gen-4 commit neuron (target -15) as the gen-5 candidate of editor claude."""
from wsp.editor import *
desc = """# gate_wrapup_L20_d15: the gen-4 "stop re-checking" neuron with a 15/9 stronger write

Edit (layer 20 MLP, neuron 5414, about 6,140 weights): identical to the carried candidate gate_wrapup_L20 (gate = the
"answer just stated" commit direction minus a threshold at its 99.8th all-token percentile, s > 10.28; up = bias proxy;
down = wrap-up minus reopen word direction mapped back through the final norm), except that the w2 column 5414 is
scaled by 15/9 (estimated logit-difference shift at commit points -15 instead of -9). w1 and w3 rows unchanged.

Purpose: a dose-response point. Selection measures the carry (-9) and this edit (-15) on the same hidden questions;
the carry scored +9.4 pp there (z 3.6, +12/-0).

Evidence (thinking mode, det1 engine, paired against the parent; trace-bank questions, parent side cached):
- DEV32 (9 cut-with-statement, 15 cut-without, 8 finished; used to compare doses): 8 -> 9 (+3.1, SE 7.1; +3/-2); cut 24 -> 19;
  the carry on the same DEV32: 8 -> 9 (+3.1, SE 5.5; +2/-1). -15 against the carry directly: +3/-3.
- HOLDOUT32 (32 other bank questions, fixed-seed draw, measured once): 17 -> 16 (-3.1, SE 3.1; +0/-1); cut 13 -> 14.
- Combined 64: +0.0 pp. I expect no gain over the carry; the stronger push did not add measurable benefit.
Note: no trace stays byte-identical (first difference at a median of 300-550 characters), so for this edit pairing behaves
like independent samples and small tests have little power.
"""
print(submit_candidate("gate_wrapup_L20_d15", desc))
