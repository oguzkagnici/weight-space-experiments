# Generation 12 report

- Created: 2026-11-14 20:20 UTC
- Parent: `gen008_carry_stack_shortans_L23_finalist_L22`
- Editors: claude, ed_f; budget 8 passes each; up to 1 candidates each
- Hidden selection subset: 256 questions (seed `selection-gen-12`)

## Suggested starting points

- **claude**: LATENT KNOWLEDGE THAT IS LOST ("knows but does not say"). Generations 10-11 showed visible reasoning habits are symptoms; look instead for cases where the network internally prefers the RIGHT option and a specific later component overrides it. Concretely: on MIXED and WRONG traces, take forced-answer probes ('So the answer is (') at several points of the thinking and read them through the layers (logit lens on the residual stream at L12-L23 projected onto the option letters, or per-layer direct attribution). Look for a systematic pattern: gold ranked higher at an intermediate layer than at the output, and one or a few late components (an MLP neuron group, a head, a direction) that push it down, e.g. by favouring the option already named most often, the familiar-sounding one, or the model's pre-thinking prior. If such an override exists and is content-independent, weaken it and show on HELD-OUT prefixes (not used to fit) that P(gold) rises before spending generate passes. If no override exists (gold is never preferred internally on wrong traces), say so clearly: that is an important negative result.
- **ed_f**: OPEN CHOICE. You are a new editor this generation (no earlier reports of your own). Read the history in this file first: four mechanism-targeted edits in generations 10-11 were real but did not move accuracy, and the noise floor is about +/-4 points. Pick the target you judge most likely to raise P(gold) on held-out probes, and justify the choice in notes.md before spending passes. Candidates the other editors suggested but did not test: (a) replace a hedging or exam-gaming paragraph with a neutral content paragraph in fixed traces and see if P(gold) moves (cause vs symptom); (b) act at the SOURCE of the generic 'first item of a list' feature (the residual direction written at list items) rather than in one head; (c) a gate on uncertainty that steers toward re-deriving content ('Let us recall the definition...') rather than suppressing words; (d) anything else the evidence points to, e.g. option-content comparison errors on two-finalist questions, or the 21 never-right questions if some of them are reasoning slips rather than missing knowledge. Apply the gate rule: show P(gold) rises on held-out prefixes before you generate.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 100 | read 100 texts | 1576.4 |
| claude | 2 | read | original | 422 | read 422 texts | 1281.2 |
| claude | 3 | read | edited | 404 | read 404 texts | 1509.5 |
| claude | 4 | generate | edited | None | read None texts | 2491.7 |
| claude | 5 | generate | edited | None | read None texts | 2402.5 |
| claude | 6 | generate | edited | None | read None texts | 1459.4 |
| claude | 7 | generate | edited | None | read None texts | 892.3 |
| claude | 8 | generate | edited | None | read None texts | 2246.7 |
| ed_f | 1 | read | original | 160 | read 160 texts | 1089.4 |
| ed_f | 2 | read | original | 22 | read 22 texts | 34.8 |
| ed_f | 3 | read | edited | 160 | read 160 texts | 979.1 |
| ed_f | 4 | score | both | 256 | paired -0.78 (SE 2.6), +21/-23, dlogp +0.1222 | 733.4 |
| ed_f | 5 | read | edited | 160 | read 160 texts | 969.0 |
| ed_f | 6 | generate | edited | None | read None texts | 2853.3 |
| ed_f | 7 | generate | edited | None | read None texts | 2337.4 |
| ed_f | 8 | generate | edited | None | read None texts | 1093.7 |

## Candidates

### claude/claude_g12_readout_letter_debias_L23

- Tensors: model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 12202

claude_g12_readout_letter_debias_L23: remove the content-free letter preference of the model's FORCED READOUTS inside reasoning (pre-thinking prior and "so far, the most likely answer is (" states).
WHAT WAS CHANGED: L23 feed_forward.w2 columns 7528, 6877, 4830, 1228, 4441, 2650 (neurons reliably active at forced-answer readouts; 7528/6877 are silent at final-answer copy positions) += share_j * v, share_j proportional to the neuron's mean activation at prior probes. v = minimal-norm pre-norm MLP-output vector giving a centered letter-logit shift of -c at the prior probes' mean norms (strength 1.0), c_L = log(mean P_L) - log(gold share L) measured on 320 pre-thinking prior probes of practice questions outside all banks/holdouts: c = [.74,.71,.72,.33,.90,-.20,-.43,-.81,-.65,-1.32] for A..J (the prior predicts A 87x vs gold 46x, J 2x vs 21x). 368 entries exceed their Q8_0 row-block absmax (max 1.9x).
WHY: generation-12 diagnosis found NO latent gold anywhere at the readout: on wrong conclusions (20 mixed-question samples + 28 other wrong traces, probes at 35/50/70/100% of the thinking) gold is the letter-lens top at no layer 14-24 (<= 2 of 55-ish probes), P10(gold) <= .07; the readout copies the trace's stated belief faithfully (L22 H12 attends the pick's option line 3-5x the gold line). The one place a read-level P(gold) gain exists is the content-free letter bias of the readout, which is also the early latent belief that wrong samples anchor on (generation 11).
GATE (read pass, edited vs parent, same texts): 64 held-out pre-thinking prior probes (passk questions, not used to fit c): P(gold) .282 -> .313 (paired +.031, SE .012), accuracy 26.6 -> 32.8% (+5/-1); forced probes after 0-800 generated tokens of mixed-question samples (315): +.068 (SE .006), +25/-4 (overstated: these golds skew late); 35% probes (25): +.070 (SE .026), +3/-0. Off-target: top-1 token changed at 1.85% of teacher-forced positions; actual next-token log-prob -0.001 on average.
EVIDENCE ON CONCLUSIONS: DEV (6 mixed passk q x 4, 8,192): +20.8 pts (SE 13.4, +8/-3), both-finish +20.0 (SE 14.5, +4/-1). HOLDOUT48 at 8,192 tokens: overall 62.5 -> 62.5%, +0.00 pts (SE 5.16, +3/-3); BOTH-FINISH n 39: +0.00 pts (SE 5.20, +2/-2); cut 7 -> 7. 3,072-token HOLDOUT BANK (96): overall +0.00 (SE 3.63, +6/-6); both-finish n 45 +2.22 (SE 2.22, +1/-0); cut 42 -> 40.
EXPECTATION: about 0 at selection. The readout-level gain is real but conclusions are formed in the reasoning text, which this edit does not improve.

### ed_f/ed_f_g12_key_ordinal_slope_removal

- Tensors: model.layers.10.self_attn.k_proj.weight, model.layers.12.self_attn.k_proj.weight, model.layers.14.self_attn.k_proj.weight, model.layers.16.self_attn.k_proj.weight, model.layers.18.self_attn.k_proj.weight, model.layers.20.self_attn.k_proj.weight, model.layers.22.self_attn.k_proj.weight
- Weights changed (bf16): 5410326

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

## Selection (hidden questions) and decision

Parent accuracy on the 256 selection questions: **60.55%**. Rule: z >= 2.0; both-finish z >= 1.0; gsm8k_boxed guard z > -2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| claude/claude_g12_readout_letter_debias_L23 | 60.16 | -0.39 | 2.1 | -0.19 | 14 | 15 | +0.0000 | 12202 |
| ed_f/ed_f_g12_key_ordinal_slope_removal | 60.16 | -0.39 | 2.3 | -0.17 | 17 | 18 | +0.0000 | 5410326 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

# Report: editor claude, generation 12

## What the model gets wrong
I tested "knows but does not say": does the network prefer the right option somewhere inside, with a later component overriding it?
- **It does not.**
  - **Setup:** forced-answer probes ("So far, the most likely answer is (") at 35%, 50%, 70% and the end of the thinking.
    - 20 wrong samples of the 15 mixed questions (pass 1).
    - 28 wrong traces of 13 other questions (pass 2).
    - Right-trace controls.
  - **Letter lens, residual h14–h24:** when the sample ends wrong, gold is never the top letter at any layer (≤ 2 of ~55 probes at any single layer), and P10(gold) ≤ .07. In right traces the lens picks gold 57–80% from h15 on.
  - **Attribution:** the only components pushing toward gold in wrong probes are the generic negative-feedback ones (L22 MLP leader suppression, L22 H11/H13/H14).
- **The readout copies the trace's stated belief.**
  - In thinking mode the current belief is readable from h15 (after L14). In direct mode it appears only at L22.
  - At the probes, L22 H12 attends to the pick's option line 3–5× more than to the gold line, and to letters the trace has already stated.
  - So wrong conclusions are wrong all the way through. This agrees with generations 2 and 10 ("missing knowledge").
- **Smaller checks:**
  - Questions where the pre-thinking prior is right but thinking ends wrong (6) are mostly lost to cuts (3 of 4 or 4 of 4 samples cut at 8,192), not to content.
  - Negation questions are rare (6%).
  - Stated hypotheses come late; when they equal the prior, the result is no worse.
- **The one measurable lever is the readout's letter bias.** The pre-thinking prior (forced answer right after `<think>`) is 29.7% accurate on 320 practice questions and badly letter-biased: it predicts A 87× against 46 golds, and J 2× against 21. Gen 11 found that wrong samples' early latent belief anchors on this prior.

## What I changed and why
- I added a constant per-letter correction to the forced readouts (`claude_g12_readout_letter_debias_L23`). It is carried by six L23 neurons that are reliably active at those readouts; n7528 and n6877 are silent at final-answer copy positions.
- Correction: c_L = log(mean P_L) − log(gold share), at full strength. It is written as the minimal-norm w2-column change that shifts the centered letter logits by −c at the readout.
- **Gate (read pass, held-out, paired against the parent):**
  - 64 pre-thinking prior probes (not used to fit c): P(gold) .282 → .313 (+.031, SE .012); accuracy 26.6 → 32.8% (+5/−1).
  - Mid-trace probes on mixed questions: +.068 (SE .006), +25/−4. This is overstated, because these questions' golds skew to late letters.
  - Off-target: top-1 changed at 1.85% of teacher-forced positions; actual-token log-prob −0.001 on average.

## Evidence on conclusions

| Set | Overall paired diff (SE), +/− | Both finished |
|---|---|---|
| DEV (6 mixed questions × 4 samples, 8,192) | +20.8 (13.4), +8/−3 | +20.0 (14.5), +4/−1 |
| **HOLDOUT48, 8,192** | **+0.00 (5.16), +3/−3** | **n 39, +0.00 (5.20), +2/−2** |
| 3,072 holdout bank (96) | +0.00 (3.63), +6/−6 | n 45, +2.22 (2.22), +1/−0 |

- Final letters stay consistent with the reasoning text (I checked the changed answers).
- Expected selection result: about 0. The readout gain is real, but conclusions are produced in the reasoning text, and this edit does not change that text's quality.

## What I would try next
1. **Stop targeting the answer path.** Four generations of evidence (gens 2, 10, 11 and 12) say the readout is faithful and wrong conclusions carry no latent gold. Any gain has to change what the reasoning text says.
2. **Cut answers on prior-right questions.** These are the clearest "knew it, lost it" cases, and they are lost by running out of budget, not by content. A content-aware loop breaker (detect repeating re-derivations and conclude with the current leader) would act exactly there. Its selection gain would mostly appear in the overall diff, not on questions both models finish, so it needs the selection rule's view on cuts.
3. **Pass@k vs single-sample variance** (majority vote +5.7 points) is the remaining content headroom. Read passes cannot reach it, because conclusions form gradually in the text. It needs generate-pass designs with more samples per question than 8 passes allow.

### ed_f

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

