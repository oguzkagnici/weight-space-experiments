# ed_a_g10_letter_ramp_removal: remove the content-free early-letter prior from 11 prior-writer neurons

TARGETED BEHAVIOUR. Wrong finished answers name two finalists more often than right ones (42/83 vs 36/180, passk 8,192 bank + 3,072 bank).
When gold and the pick are both in the last stated pair, the model is right 15/15 when gold is the EARLIER letter and 14/27 when it is the LATER one
(wrong in 9 questions). Hypothesis: the early-letter prior known from direct mode (near-constant writers L20 n1000, L17 n616/n2401, L23 n4597 ...)
tilts the finalist choice.

CHANGE (w2 columns only, 6 tensors): L15 n4408, L17 n616, n2401, L19 n6621, n6612, L20 n1000, n3964, n3142, L22 n2820, n2389, L23 n4597.
They were picked by direct logit attribution at 82 forced-commit probes ("Thus I'll go with (" after a two-finalist statement) as consistent writers
of a linear A->J ramp in the letter logits. For each column: w2[:,j] -= 0.65 * slope_j * u_j, where u_j is the minimal-norm residual direction
(weighted by Q8_0 block headroom, entries kept under 0.98x block absmax) whose read-out through post_ffn_norm gain, final-norm gain and the centred
letter embeddings is the pure ramp. Each column keeps 35% of its ramp; nothing else in its letter read-out changes. No training, no fitting.

MECHANISM CHECK (read pass, 82 probes): ramp in non-finalist letter logits -0.195 -> -0.060 per letter step, BUT pair decisions are unchanged
(T0.6 P(gold within pair) .662 -> .663; top letter changed in 0/82). The finalist preference is not the letter-label prior. A swap test
(original model, 53 pair statements) shows it is MENTION ORDER: naming a finalist first is worth ~1.2 logits toward it.

MEASURED (generate, edited only, cap 8,192, paired against the parent banks):
- DEV, 6 mixed/finalist-loss passk questions x 4 samples: 37.5 -> 37.5 (+0.0, SE 13.5, +5/-5); both-finish +5.0 (SE 15.3, +5/-4).
- HOLDOUT48 bank: 62.5 -> 64.6, +2.1 (SE 5.6, +4/-3); both-finish n39 +0.0 (SE 5.2, +2/-2); cut 7 -> 7.
- 1/48 holdout traces byte-identical; median first difference at 3% of the trace: the swaps are resampling, not the mechanism.
Expect about 0 in selection. Submitted because it is the only edit measured on the holdout bank.