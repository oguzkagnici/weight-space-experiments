# Report: editor ed_f, generation 12

**Candidate submitted:** `ed_f_g12_key_ordinal_slope_removal` (7 tensors, 5.4M weights). All 8 passes used: 4 read, 1 score, 3 generate.
**Holdout48 at 8,192 tokens:** overall +2.08 points (SE 6.31, +5/-4), 62.5 -> 64.6%. Both-finish (39 questions): +2.56 (SE 5.79, +3/-2), 71.8 -> 74.4%.
**Expected on selection:** about +1 to +3 points, probably below the bar. GSM8K was not tested and is a real risk (the edit changes attention keys on every token).

## What the model gets wrong: the answer depends on where the gold option stands
- **Late gold options lose.** Parent traces at 8,192 tokens (passk + bank, 243 ten-option traces, 85 questions):

  | gold position | right | wrong (finished) | cut | n |
  |---|---|---|---|---|
  | A-E | 76% | 9% | 15% | 119 |
  | F-J | 44% | 34% | 22% | 124 |

  Question-level correlation between gold index and finished accuracy: r = -0.235 (permutation p .017; -0.33 after removing category means). Finished traces are also longer when gold is late (3,116 vs 2,066 tokens). Gold index is uniform in the practice set.
- **It replicates on untouched questions.** The parent's holdout48 bank (read only after the edit was fixed): gold A-E 80.8% (n 26), gold F-J 40.9% (n 22).
- **It is not coverage.** Every letter is discussed in 81-88% of traces and the gold option in 96-100% of wrong traces. Wrong picks are not earlier than chance given the gold position. An early gold is found and kept; a late gold loses to an earlier incumbent. This is the same family as the mention-order effect (gens 10-11) and direct-mode option primacy (gens 2-3).
- **Other checks (free, from the banks):**
  - Content-only headroom from trajectory variance is small: finished accuracy 62.2%, majority-of-finished 65.3% (+3.1). The +5.7 "maj@4" figure includes cut-avoidance.
  - Wrong finished answers are in verbal subjects (law, health, other, economics, psychology). The 17 never-right questions give the same wrong letter in most samples: stable wrong beliefs, not slips.
- **Dead ends (pass 1, 160 forced-answer probes after thinking prefixes):**
  - No intermediate layer prefers gold more than the output (logit-lens accuracy .24 at L16, .26 at L22, .32 at L24).
  - Dropping low-energy directions of late MLP down-projections (rank reduction) does nothing: +1 to +4 probes of 160, inconsistent across splits. The spectra are flat.
  - A constant per-letter offset improves log P(gold) by +0.3 to +0.65 but not accuracy (-3 / +3 probes), as in direct mode.

## What I changed and why
- **Mechanism found (pass 2):** the key inputs of option-line tokens carry a generic ordinal direction in every attention layer. Slot means fall A > B > ... > J along it (34-59% of slot-mean variance; split-half cosine .92-.96; A-to-J gap 0.6-1.8 against a within-slot sd of 0.2-0.4). `k_proj` reads it 1.5-2x more strongly than a random direction, so every option-reading head can prefer early items. Earlier generations removed the preference only in L22/L20 queries.
- **Edit:** in `k_proj` of layers 10, 12, 14, 16, 18, 20 and 22, `W_k <- W_k (I - u u^T)`, with u the unit ordinal-slope direction of that layer's slot means (from 22 prompts outside all banks). Rank 1 of 2,048 per layer, 2.9-3.8% relative change. Queries and values are untouched.

## Evidence
- **Gate (read pass, 160 held-out probes, forced "So the answer is ("):**

  | probes | accuracy | +/- | P(gold) change (SE) |
  |---|---|---|---|
  | all 160 | 31.9 -> 40.0% | +14/-1 | +.027 (.013) |
  | gold F-J (77) | 24.7 -> 40.3% | +12/-0 | +.095 (.021) |
  | gold A-E (83) | 38.6 -> 39.8% | +2/-1 | -.035 (.014) |

  Log P(gold) +.48 (SE .12); gold rank 2.94 -> 2.79; predicted "A" 44 -> 17 (gold A: 7). A constant letter offset does not do this.
- **Rank-2 variant (slope + first-item direction):** +10/-3 at the probes. In direct mode (score, 256 questions) it was purely redistributive: -0.8 points (SE 2.6), A 38 -> 26%, B-E 46 -> 37%, F-J 33 -> 44%, and it over-corrected option A. I dropped it for the rank-1 edit, which was not scored in direct mode.
- **Holdout48 at 8,192 (final):**
  - Overall +2.08 (SE 6.31, +5/-4); both-finish +2.56 (SE 5.79, +3/-2); cuts 7 -> 7; 8 of 39 both-finish answers changed.
  - Gold F-J: 40.9 -> 59.1% (+5/-1). Gold A-E: 80.8 -> 69.2% (+0/-3).
  - No trace is identical to the parent's (first difference at a median 0.8% of the trace).
- **DEV (6 passk questions x 4 samples, chosen before the run; picked on parent failures, so inflated):** 54.2 -> 87.5% (+9/-1). Late-gold questions went BBBF -> BFFF, HDDH -> HDHD, CEGG -> GGGG and BDDB -> IIII (a never-right question right in 4 of 4). Early-gold controls BBBB and AAAA were unchanged.
- **Reading:** this is the first edit of gens 10-12 that changes conclusions on both-finish questions in the predicted direction. But in generation it is partly a transfer from early-gold to late-gold questions, and the net (+2 to +3) is far inside the noise of 48 questions.

## What I would try next
1. **Dose.** The loss on early-gold questions suggests the projection is too strong. Try 0.5-0.7 of the projection, or fewer layers, and pick the dose by the early-gold/late-gold split on generated answers rather than by probes.
2. **Measure by gold position.** Report every candidate's selection result split by gold A-E vs F-J; the parent's gap (about 35-40 points) is the largest content effect found so far and is cheap to track.
3. **Find what the incumbent advantage costs.** A symmetric model should gain from removing a position bias; test on mixed questions whether late-gold losses come from an early option being accepted before the gold is read (probe the belief after each option's verdict).
4. **Check GSM8K and direct mode for the submitted rank-1 edit** before carrying it; neither was measured.
