# Submit Unit A (holdout-measured; it is the only built edit) as the gen-9 candidate of editor claude.
from wsp.editor import *
DESC = """Unit A: close the thinking right after a firm single-letter conclusion (L23, one repurposed neuron).

CHANGED: model.layers.23.feed_forward.{w1,w3,w2} at neuron 5217 only (one of the quietest L23 neurons: mean |out| 0.028). All other entries are identical to the parent.
- Gate row: w1 = K*(d - th*mu/m), with K 2 and threshold 8.
  - d = unit mean difference at the L23 MLP input: tokens that end a single-letter answer statement inside the thinking ("... answer is (E)." / "I'll go with J.") minus all other tokens.
  - mu/m = mean think-token input, so its dot product c is about 1 (a bias proxy).
  - Leave-one-text-out AUC of d is 0.988 (at L20).
- Up row: w3 = mu/m.
- Write column: w2 = E[</think>] - mean E[reopen words: But, However, Wait, Let, We, Alternatively, ...]. It is mapped through the final norm gain and inverted through post_ffn_norm, at scale 1, and each entry is clipped to 0.98 x the absmax of its Q8_0 block.

TARGET BEHAVIOUR: in cut traces (3,072-token cap, scored 0) the model often states a firm answer ("Thus answer should be (I).") and then reopens ("But let's double-check ..."). It re-deliberates until it hits the cap.
- In finished traces, the first firm conclusion equals the final answer in 38 of 39.
- On read-pass recordings the unit fires on 72% of statement ends, raising P(</think>) from a median of 0.000 to 0.999 there. Only 72 of 69k other tokens change any of the tracked logits (</think> and the reopen words) by more than 0.5, and all 72 are conclusion-like. No effect on prompt, two-letter ("B or D?") or post-think tokens.
- It is not a length trigger: it fires only where an answer has just been stated.

EVIDENCE (generate, det1 Q8_0, cap 3,072, edited only, paired against the parent's byte-identical trace banks):
- HOLDOUT BANK (96 ids, never used to fit): parent 46 -> edited 48 correct, +2.1 pp (SE 3.9, z 0.53), +8/-6; cut 42 -> 38.
  - Halves: +0.0 (SE 5.2, +3/-3) and +4.2 (SE 5.9, +5/-3).
  - 11 of the 42 cut holdout traces now close right after a conclusion, and 7 of those 11 are right. Every rescue closes after "Thus/Therefore ... (X)."
  - Losses: 7 finished traces were resampled into new cuts (5 of them right before), and 1 closed on a wrong conclusion.
- Main trace bank (96; the gate direction was fit on texts from bank traces, so this is not fully out of sample): +8.3 pp (SE 3.2, z 2.60), +9/-1, cut 47 -> 41.
- Removing n5217's original output resamples ~92% of traces (median first divergence at 15% of the trace). That resampling noise (about ±6 swaps per 96) is comparable to the mechanism's net effect, so the true gain is probably small (+2 to +4 pp)."""
print(submit_candidate("claude_g9_firm_conclusion_close_L23", DESC))
