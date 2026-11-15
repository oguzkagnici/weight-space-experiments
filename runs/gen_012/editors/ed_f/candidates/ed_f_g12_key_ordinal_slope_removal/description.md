# ed_f_g12_key_ordinal_slope_removal: remove the ordinal "list position" direction from the attention KEYS of layers 10-22 (option primacy)

TARGETED BEHAVIOUR (content). The parent's conclusion depends on WHERE the gold option stands in the list. Parent traces at 8,192 tokens
(passk + bank, 243 ten-option traces, 85 questions): gold A-E -> right 76%, wrong 9%, cut 15%; gold F-J -> right 44%, wrong 34%, cut 22%
(question-level r = -0.235 between gold index and finished accuracy, permutation p .017; gold index is uniform in the data). Replicated on the
untouched holdout48 bank (parent): gold A-E 80.8% (n 26) vs gold F-J 40.9% (n 22). It is not coverage (every letter is discussed in 81-88% of traces,
gold in 96-100% of wrong traces): an early gold is found and kept, a late gold loses to an earlier incumbent. Same family as the mention-order effect
(gens 10-11) and the direct-mode option primacy (gens 2-3), which were only treated in the L22/L20 queries.

MECHANISM / EVIDENCE. Key inputs (k_proj inputs) of option-line tokens carry a generic ordinal direction in EVERY attention layer: slot means line up
A > B > ... > J (PC1 = 34-59% of slot-mean variance; split-half cosine of the slope direction .92-.96; A-to-J gap 0.6-1.8 vs within-slot sd 0.2-0.4;
22 prompts of questions outside all banks). k_proj reads it 1.5-2x more strongly than a random direction.

CHANGE (7 tensors, model.layers.{10,12,14,16,18,20,22}.self_attn.k_proj.weight; about 5.5M entries differ after bf16 rounding; relative change 2.9-3.8% each):
W_k <- W_k (I - u u^T), u = unit ordinal-slope direction of that layer's slot means (sum_k (k - 4.5) * mean key input of slot k). Rank 1 of 2,048 per layer;
queries, values and all other weights untouched. No fitting: u is a weighted difference of means of recorded activations.

GATE (held-out P(gold) read pass, pass 5): 160 forced "So the answer is (" probes after thinking prefixes of parent traces (96 bank traces at <=40% of the
thinking, 64 passk traces at <=15%); none of these questions was used for u. Forced-answer accuracy 31.9 -> 40.0% (+14/-1); P(gold) .317 -> .345
(+.027, SE .013); log P(gold) -4.13 -> -3.65 (+.48, SE .12); gold rank 2.94 -> 2.79. Gold F-J (77 probes): 24.7 -> 40.3% (+12/-0), P(gold) +.095 (SE .021);
gold A-E (83): 38.6 -> 39.8% (+2/-1), P(gold) -.035 (SE .014). Predicted "A" 44 -> 17 of 160 (gold A: 7).
(A rank-2 variant, slope + first-item direction: +10/-3 at the probes; in DIRECT mode on 256 questions it was purely redistributive, -0.8 (SE 2.6), A 38 -> 26%,
B-E 46 -> 37%, F-J 33 -> 44%, and over-corrected option A; the submitted rank-1 edit was not scored in direct mode.)

HOLDOUT48 at 8,192 tokens (final measurement, edited-only passes 6-7, paired with the parent bank):
- overall: 62.5 -> 64.6, +2.08 points (SE 6.31), +5/-4; cut 7 -> 7; mean tokens 3,472 -> 3,368.
- BOTH-FINISH (39 questions): 71.8 -> 74.4, +2.56 points (SE 5.79), +3/-2; 8 of 39 answers changed.
- by gold position: gold F-J (22) 40.9 -> 59.1 (+5/-1); gold A-E (26) 80.8 -> 69.2 (+0/-3). So in generation the edit moves conclusions toward late options:
  a clear gain where gold is late, a smaller loss where gold is early; the net is positive but far inside the noise.
- 0 of 48 traces identical (median first difference at 0.8% of the trace): the edit acts on every token, all traces are resampled.
DEV (pass 8; 6 passk questions x 4 samples at 8,192, chosen BEFORE the run as 4 late-gold questions with wrong parent samples + 2 early-gold controls;
selection on parent failures inflates this): 54.2 -> 87.5 (+9/-1); the late-gold questions went BBBF->BFFF, HDDH->HDHD, CEGG->GGGG, BDDB->IIII
(a never-right question became right in 4 of 4 samples); controls BBBB and AAAA unchanged.

EXPECTATION. The first edit of gens 10-12 that changes answers on both-finish questions in a predicted direction, but part of the effect is redistribution
from early-gold to late-gold questions; expected net on selection about +1 to +3 points, probably below the z >= 2 bar. Not tested: GSM8K (the edit changes
attention keys everywhere, so the transfer guard is a real risk), dose (a weaker projection may keep the late-gold gain with less early-gold loss).