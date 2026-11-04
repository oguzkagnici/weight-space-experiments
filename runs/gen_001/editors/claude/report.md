# Report of editor `claude`, generation 1

**What the model gets wrong.**
- Direct-answer accuracy on practice questions is 38% (512 DEV).
- The answer forms very late. Read through the output layer, the residual stream predicts the letter-token prior ("A") until layer 12. It reaches 29% only after layer 22's attention and 38% at the end.
- One head, L22.H12, routes the chosen option to the answer position: +3.1 logits to the correct letter on average.
- When the model is wrong, H12 itself picked the wrong option (its top letter equals the model's wrong pick in 57% of cases). That is a content and knowledge failure that I could not repair.
- The visible, repairable obstacle is a letter-position bias. On 10-option questions the final centred letter logits are A +1.05, B +1.27, C +0.70 ... I -1.53, J -1.45, while the true answers are flat.

**Mechanism.**
- The L22 MLP has one "leader suppressor" neuron per letter, at w2 columns C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310.
- Each fires when its letter leads the stream entering the MLP, pushing that letter down by 3 to 4.6 logits.
- A and B have no such neuron; A even has a booster (L22 n2212), and L23 has A/B boosters (n1685, n2595, n1439).
- Total suppression of the leader, L22 and L23 together, is A -1.2, B -2.2, and C..J -4.1 to -5.6. An early-letter leader keeps its lead and a late-letter leader is knocked down.
- The ceiling of removing the bias entirely, estimated with cross-fitted constant letter offsets on recorded outputs, is about +2 points.

**What I changed.**
- I scaled by 0.5 only the direct letter-readout component of those neurons' w2 columns: the 9-dimensional subspace that the tied letter rows read through post_ffn_norm and the final norm.
- Candidate `equalize_boost` covers the 8 suppressors and the 4 A/B boosters (24,398 weights).
- Candidate `equalize_l22` covers the 8 suppressors only (16,254 weights).

**Evidence.**
- Screening on 256 DEV questions (paired):

  | Edit | Accuracy change | SE |
  |---|---|---|
  | Suppressors, α 0.5 | +1.56 | 0.95 |
  | Suppressors, α 0 | +1.56 | 1.23 |
  | Suppressors + L23 suppressors | +1.17 | 1.41 |
  | **Suppressors + boosters** | **+2.34** | **1.35** |

- HOLDOUT, 512 practice questions never used for any choice:
  - `equalize_boost`: +1.37 (SE 1.01), 17 wrong→right / 10 right→wrong;
  - `equalize_l22`: +0.98 (SE 0.85), 12 / 7.
- Both are consistently positive but not significant on their own.
- Side effect: the correct letter's log-probability falls by 0.13–0.15. The suppressors also damp overconfidence, so a calibration-sensitive metric would get worse.

**What I would try next.**
1. The B over-preference (B predicted 49–58 of 256) is not explained by these neurons. Look for B writers before layer 21. The stream before L22 already carries A +1.18 and B +1.13.
2. Equalise *upward* instead: give A and B a suppressor of the same strength. Many new weights would have to be written, but it would keep calibration.
3. The main error is H12 choosing the wrong option. Check whether its query/key selection is biased by the option's position, using attention patterns from read passes.
