# Generation 10 report

- Created: 2026-11-12 04:12 UTC
- Parent: `gen008_carry_stack_shortans_L23_finalist_L22`
- Editors: claude, ed_a; budget 8 passes each; up to 1 candidates each
- Hidden selection subset: 256 questions (seed `selection-gen-10`)

## Suggested starting points

- **claude**: Decision points in MIXED questions (passk file: the same question answered right in some samples and wrong in others). Teacher-force right and wrong samples and probe, at regular points, what the model would answer if forced to answer now (append a short forced-answer suffix and read the option-letter probabilities), to find WHERE in the trace the eventual answer gets decided and whether the deciding step was a low-probability sampled choice that the model itself rates lower than the alternative. If decisions hinge on specific moments (naming a candidate option, accepting or rejecting an option during a sweep), look for a component that could make the model follow its own better-supported option there.
- **ed_a**: Systematic content slips. Compare the 20 finished-wrong answers on mixed questions (and finished-wrong answers on never-right questions) with right answers to the same or similar questions: is there a recurring, specific failure (e.g. losing a NOT/EXCEPT/LEAST in the question, misreading a quantity or unit, judging an option against the wrong criterion, rejecting the right option for a stated but false reason)? Find the internal correlate (a direction or component active when the slip happens or when the question contains the trigger) and repair that one failure. Check its coverage on a random sample of practice questions before investing passes.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 50 | read 50 texts | 1972.7 |
| claude | 2 | read | original | 59 | read 59 texts | 1255.1 |
| claude | 3 | read | original | 129 | read 129 texts | 1762.8 |
| claude | 4 | read | original | 52 | read 52 texts | 1583.1 |
| claude | 5 | read | edited | 24 | read 24 texts | 776.0 |
| claude | 6 | generate | edited | None | read None texts | 2627.6 |
| claude | 7 | generate | edited | None | read None texts | 1517.4 |
| claude | 8 | generate | edited | None | read None texts | 3694.3 |
| ed_a | 1 | read | original | 82 | read 82 texts | 2008.0 |
| ed_a | 2 | read | edited | 82 | read 82 texts | 1570.5 |
| ed_a | 3 | generate | edited | None | read None texts | 2801.2 |
| ed_a | 4 | read | original | 106 | read 106 texts | 1682.9 |
| ed_a | 5 | generate | edited | None | read None texts | 2993.6 |
| ed_a | 6 | generate | edited | None | read None texts | 2000.3 |
| ed_a | 7 | read | edited | 106 | read 106 texts | 2149.3 |

## Candidates

### claude/claude_g10_hedged_recheck_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 8186

# claude_g10_hedged_recheck_L23: re-check after a HEDGED commitment instead of wrapping up (content-directed, L23 n881)

TARGETED BEHAVIOUR. When the parent's first commitment inside the thinking is hedged ("I'd guess (X)", "likely (X)", "seems (X)"),
it is right only 30% of the time (20/66 finished 8,192 traces), against 74% for firm commitments (128/173). Re-checking after a
hedged first commitment raises it to 30/66 (the model switches 19 times). The parent's L20 wrap-up unit pushes wrap-up words after
ANY stated answer. This unit pushes re-check openers only after hedged commitments, leaving firm ones alone.

WHY THIS AND NOT A DECISION-POINT EDIT (diagnosis, passes 1-3, practice questions only):
- The letters at in-thinking commitments are near-deterministic (median p 1.00 in right and wrong traces), and low-probability
  sampled tokens are equally frequent in right and wrong traces (30 vs 31 per 1k): wrong answers are not unlucky samples.
- A forced answer right after a two-finalist statement equals the final pick in 51/58 traces (gold-in-pair: probe 20/31,
  realised 21/31): the conclusion is fixed before the finalists are named, so committing there changes nothing.
- Forced answers at 15/40/70% of the thinking are right in 14/17/22 of 43 traces vs 26 final: the reasoning improves the answer
  gradually; freezing an earlier belief loses.

CHANGE (layer 23 MLP only; 3 tensors):
- Freed n881 by folding it into its near-duplicate n331 (w1/w3/w2 cos >= 0.9996): w2[:,331] += 0.9893*w2[:,881].
- n881 gate (w1 row) = K*(w_lda - T*p), K=1: w_lda = shrinkage-LDA direction at the L23 MLP input, paragraph breaks right
  after a hedged commitment (49) vs breaks after firm commitments (75) + other generated tokens (~19k), from teacher-forced
  passk/trace-bank traces; projected off the first-4-prompt-token subspace (sink pre-activation <= -44). p = minimum-variance bias
  proxy (x.p mean 1.00, sd 0.022). T = 60th percentile of hedged-break scores. 8-fold grouped CV of the direction: AUC hedged vs
  firm 0.76, hedged vs other 0.987; in-sample firing: hedged 43%, firm 8%, other 0.02%.
- n881 up (w3 row) = 0.2*p. Write (w2 col) = g_final*(mean E[But, Wait, Let, Alternatively] - mean E[Thus, Hence, </think>,
  Therefore, So])/g_post, scaled so 90% of entries fit 0.98 x their Q8_0 block absmax, the rest clipped (|col| 1.05).

MECHANISM (pass 5, edited read on 24 of the fitting texts, in-sample): at hedged breaks median P(re-check openers) 0.36 -> 0.57,
P(wrap openers) 0.04 -> 0.006; firm breaks unchanged (0.27 / 0.35); top-1 token changed at 0.042% of 69k other positions
(mean |d logprob| 0.0004).

MEASURED (generate, cap 8,192, edited only, paired against the parent's byte-identical banks, same seeds):
- HOLDOUT48 (48 untouched holdout ids, measured once): 62.5 -> 66.7, +4.17 (SE 4.17), +3/-1; cut 7 -> 5; tokens 3,472 -> 3,098.
  BOTH-FINISH (40 questions): +0.00 (SE 3.58), +1/-1. 9/48 traces identical; median first difference at 12% of the trace.
  Re-check opener after a hedged commitment in the generated holdout traces: parent 11/20, edited 8/21 (not visible).
- DEV (6 passk questions whose 4 parent samples all contain hedged commitments, x4 samples; partly in-sample for the gate):
  29.2 -> 50.0, +20.8 (SE 8.5), +5/-0; both-finish +21.1 (SE 9.6), +4/-0. CAVEAT: questions chosen by a property of the parent's
  samples (hedged, so mostly wrong) and every edited trace resamples from ~9% on; regression to the mean inflates this.
VERDICT: the unselected holdout shows NO content effect (both-finish +1/-1). Expected selection outcome: fails the content check.

### ed_a/ed_a_g10_letter_ramp_removal

- Tensors: model.layers.15.feed_forward.w2.weight, model.layers.17.feed_forward.w2.weight, model.layers.19.feed_forward.w2.weight, model.layers.20.feed_forward.w2.weight, model.layers.22.feed_forward.w2.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 19098

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

## Selection (hidden questions) and decision

Parent accuracy on the 256 selection questions: **60.55%**. Rule: z >= 2.0; both-finish z >= 1.0; gsm8k_boxed guard z > -2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| ed_a/ed_a_g10_letter_ramp_removal | 63.67 | +3.12 | 2.1 | +1.49 | 19 | 11 | +0.0000 | 19098 |
| claude/claude_g10_hedged_recheck_L23 | 59.38 | -1.17 | 1.8 | -0.65 | 9 | 12 | +0.0000 | 8186 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

_(no report yet)_

### ed_a

# ed_a, generation 10: the finalist pick, letter prior vs mention order

Candidate: `ed_a_g10_letter_ramp_removal`. 7 of 8 passes used: 4 read, 3 generate at 8,192 tokens. The candidate is expected to score about 0. The useful result is a negative one about where the finalist bias comes from.

## What the model gets wrong
- **Wrong answers often come from a pick between two finalists.** Banks used: the pass@4 bank at 8,192 tokens and the 3,072 bank.
  - Wrong finished answers name two finalists ("either B or F") more often than right ones: 42 of 83 against 36 of 180.
  - Take the cases where gold and the pick are both in the last stated pair:
    - when gold is the earlier letter, the model is right 15 of 15 times (10 questions);
    - when gold is the later letter, it is right 14 of 27 times (wrong on 9 questions).
- **Things that are not the problem:**
  - Early anchoring: the first stated preference comes late (after 30% of the trace in 208 of 214 traces) and is the final answer.
  - A letter tie at the commit token: after the pair line, a forced "Thus I'll go with (" is already nearly decided. Only about 6 of 82 probes are near-ties.
- **The bias is mention order, not the letter label.**
  - Read pass 1 (82 probes): the letter logits do carry a content-free ramp from A to J. It is −0.195 per letter step on the non-finalist letters, about 1.75 logits from A to J.
  - Read pass 2: removing most of that ramp (down to −0.06) left the pair decisions unchanged. P(gold within the pair) went from .662 to .663, and the top letter changed in 0 of 82 probes.
  - Read pass 4 is the swap test: on 53 pair statements, I swapped which finalist is named first and forced a commit right after that sentence.
    - Naming an option first is worth about 1.2 logits towards it (median 0.9), which is about 2 logits at temperature 0.6.
    - Content still dominates: 4 of 53 picks flip.
    - The model also tends to name its favourite first. When the first-named finalist is the later letter, its preference is +4.2 logits (n 18), against +2.4 when it is the earlier letter.
  - Pass 7: four L20 heads (9, 12, 13, 15) attend more to the first-named letter, and that correlates .63–.75 with the preference. But zeroing them did not reduce the order effect (1.20 → 1.38 logits), so they are not its cause.

## What I changed (candidate)
- **Target:** the 11 neurons that consistently write the A→J letter ramp, identified by direct logit attribution at the probes: L15 n4408, L17 n616 and n2401, L19 n6621 and n6612, L20 n1000, n3964 and n3142, L22 n2820 and n2389, L23 n4597. Several of them are the prior writers found in phase 1.
- **Edit:** for each w2 column, remove 65% of its ramp along a minimal-norm direction weighted by Q8_0 block headroom. Entries stay under 0.98× the block absmax. The edit changes 19,098 weights in 6 tensors, and the rest of each column's letter read-out is unchanged.

## Evidence (generate, edited only, cap 8,192, paired against the parent banks)
| set | parent → edited | diff (SE) | +/− | both finished: diff (SE), +/− | cut |
|---|---|---|---|---|---|
| DEV: 6 finalist-loss pass@4 questions × 4 samples | 37.5 → 37.5 | +0.0 (13.5) | +5/−5 | +5.0 (15.3), +5/−4 | 3 → 2 |
| **HOLDOUT48 bank** | 62.5 → 64.6 | **+2.1 (5.6)** | +4/−3 | **+0.0 (5.2), +2/−2 (n 39)** | 7 → 7 |

- On the holdout, 1 of 48 traces is byte-identical, and traces first differ at a median of 3% of their length.
- The edit moves letter logits wherever letters occur, so every trace is resampled almost at once.
- The swaps are therefore noise, not the mechanism; pass 2 showed the decisions themselves do not change. Expect about 0 in selection.

## What I would try next
1. **The mention-order primacy is real and causal (about 1.2 logits), but I have not localised it.** The L20 heads that correlate with it are not the cause. Next steps:
   - Run the swap probes (code/016, work/swap_sel.json) with `attn_last` on all attention layers.
   - Ablate candidate heads or the conv layers one layer at a time. Short-convolution layers can carry recency or primacy along a sentence.
2. **Do not spend passes on letter-label debiasing in thinking mode.** The decision is made in the text before any letter token is chosen. The ramp only matters at real near-ties, which are rare.
3. **Plan around the queue.** A pass at 8,192 tokens took 35–50 min of compute, plus hours of waiting in the queue. Queue the holdout passes early: snapshots are taken when a pass is queued, so the edit can be changed afterwards.

