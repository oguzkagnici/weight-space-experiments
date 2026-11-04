# submit the two candidates chosen on DEV (rebuilt exactly by code/003)
import subprocess, sys
from wsp.editor import *
def build(f): subprocess.run([sys.executable, "code/003_amp_heads.py", str(f)], check=True, capture_output=True)
build(1.5)
print(submit_candidate("routers_x1p5", """Edit: scale the outputs of three weak "early router" heads by 1.5: model.layers.14.self_attn.o_proj.weight columns of head 15 (15*128..16*128-1) and head 12 (12*128..13*128-1), and model.layers.20.self_attn.o_proj.weight columns of head 7 (7*128..8*128-1). 786,432 weights.
Mechanism (H5): besides the main answer-routing head L22.H12 (own-choice accuracy .371), these heads write a weak, partly independent vote for an option (own-choice accuracy .26-.28, agreeing with H12 only 31-36%, right on 15-16% of H12's mistakes; direct effect ~0.1 logit). Amplifying them gives the ensemble more say.
Evidence (paired, chosen on DEV): DEV screen256 +1.17 (SE 0.87), DEV screen2 +1.17 (SE 1.30); dose-response x1.5 +1.17, x2 +1.95, x3 +1.95 (log-prob cost grows: -0.15, -0.20, -0.55). HOLDOUT (measured once, never used to choose): 768 questions, about +0.78 (SE ~0.55): HOLDOUT[0:512] +0.59 (SE 0.70) +8/-5, HOLDOUT[512:768] +1.17 (SE 0.87) +4/-1. The correct letter's log-prob falls by ~0.14."""))
build(2)
print(submit_candidate("routers_x2", """Edit: the same three heads (L14.H15, L14.H12, L20.H7; o_proj input columns of each head) scaled by 2. 786,432 weights.
Evidence: DEV screen2 +1.95 (SE 1.51), +10/-5. HOLDOUT[0:512] (once): +0.20 (SE 1.09), +16/-15, dlogp -0.303. Submitted as the stronger dose; HOLDOUT shows it is about null."""))
