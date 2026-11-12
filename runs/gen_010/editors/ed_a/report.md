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
