# submit the candidates (rebuilt exactly by code/004)
import subprocess, sys
from wsp.editor import *
def build(f): subprocess.run([sys.executable, "code/004_l12h13.py", str(f)], check=True, capture_output=True)
build(0.5)
print(submit_candidate("l12h13_half", """Edit: scale the output of attention head L12.H13 by 0.5 (input columns 13*128..14*128-1 of model.layers.12.self_attn.o_proj.weight). 262,144 weights.
Mechanism: L12.H13 is the main attention writer of the option-correctness belief at the option-end tokens (generation 2, ed_b); its strength sets the CONFIDENCE of the final answer more than its ranking. Halving it is a dose the generation-2 search never tested (x0 and x2 were tested).
Evidence (paired; chosen on DEV): DEV 512 +0.39 (SE 0.87), gold rank -0.002 (SE 0.027), dlogp +0.19. HOLDOUT 768 (measured once per question, never used for choosing): +0.39 (SE 0.65), +14/-11; gold rank -0.048 (SE 0.026); dlogp(correct) +0.262 (SE 0.020). Expect ~0 accuracy and a robust calibration gain."""))
build(0)
print(submit_candidate("l12h13_ablate_retest", """Edit: ablate L12.H13 (its 128 o_proj input columns set to 0), rebuilt from the published description of generation-2 candidate ed_b/ed_b_g2_L12H13_ablate. 262,144 weights.
Purpose: an unbiased re-test, on a fresh hidden subset, of the best hidden candidate of generation 2 (there +1.25, SE 1.20, z 1.04; gold rank -0.11; dlogp +0.50), whose estimate is inflated by being the best of five.
My evidence: DEV 512 -1.37 (SE 1.45), gold rank +0.076 (SE 0.057), dlogp +0.40; HOLDOUT 512 -0.20 (SE 1.40), +25/-26, gold rank -0.104 (SE 0.058), dlogp +0.535 (SE 0.054)."""))
