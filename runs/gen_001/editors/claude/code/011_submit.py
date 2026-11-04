# submit the two candidates (rebuilt from code/010 to be exact)
import subprocess, sys
from wsp.editor import *
def build(args):
    subprocess.run([sys.executable, "code/010_edit_suppressors.py", *args], check=True, capture_output=True)
build(["0.5", "none", "0.5"])
print(submit_candidate("equalize_boost", """Edit: scale the DIRECT LETTER READOUT (the component of a w2 column that the tied letter rows read through post_ffn_norm and the final norm; a 9-dim subspace) by 0.5 for (a) the 8 letter-specific "leader suppressor" neurons of layer 22's MLP, model.layers.22.feed_forward.w2.weight columns C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310, and (b) 4 early-letter "booster" neurons: L22 column 2212 (A), L23 columns 1685 (A), 2595 (A), 1439 (B). The rest of each column is unchanged. 24,398 weights differ in bf16.
Mechanism: after head L22.H12 moves the chosen option to the answer position, the L22 MLP suppresses the currently leading letter through one neuron per letter (-3 to -4.6 logits when that letter leads). There is no such neuron for A or B, and A/B have boosters. Total suppression of a leader: A -1.2, B -2.2, C..J -4.1 to -5.6. That asymmetry makes the model over-pick early letters (practice: A..C predicted 249 vs 168 true out of 512). Halving the late-letter suppressors and the A/B boosters equalises it.
Evidence (paired, practice): screen256 DEV +2.34 (SE 1.35), +9/-3; dose-response for the suppressors alone: alpha 0.5 +1.56 (SE 0.95), alpha 0 +1.56 (SE 1.23) with more churn; adding layer 23's suppressors hurt (+1.17, SE 1.41). HOLDOUT 512 (never used for choices): parent 32.62 -> 33.98, +1.37 (SE 1.01), +17/-10. Side effect: the correct letter's log-probability falls (-0.13), because the suppressors also damp overconfidence."""))
build(["0.5"])
print(submit_candidate("equalize_l22", """Edit: the same as equalize_boost but only part (a): the direct letter readout of the 8 layer-22 late-letter leader suppressors (model.layers.22.feed_forward.w2.weight columns 1812, 464, 7357, 6492, 3344, 397, 2786, 3310) is scaled by 0.5. 16,254 weights differ in bf16.
Evidence: screen256 DEV +1.56 (SE 0.95), +5/-1; HOLDOUT 512: +0.98 (SE 0.85), +12/-7, dlogp(correct) -0.153."""))
