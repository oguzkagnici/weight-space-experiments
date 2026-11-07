# Report of editor `claude`, generation 6

## What the model gets wrong
- **Law and health questions are cut at the cap 18 of 23 times.** These traces do not do a categorically different thing from finished ones: they sweep the options ~3 times instead of ~1.2, but per token the sweeping and "Let's examine each option ..." restarts are only slightly denser (0.21 vs 0.16 per 1k tokens). Mostly the model simply does not know the fact and keeps deliberating.
- **A specific, fixable failure inside that group: oscillating between two finalists.** 18 of the parent's 53 cut bank traces narrow the choice to two options ("either F or G", "between C and H"), on average 58% into the trace, then spend ~1,300 tokens re-sweeping without deciding. Finished traces rarely narrow (4 of 43): they go straight to an answer.
- **Deciding there would usually beat being cut** (pass 1, forced answer after the first two-finalist paragraph):
  - correct in 8 of 18 cut traces; 7 of 11 when gold is one of the two (probability on the pair 0.99);
  - forcing at the cap instead gives 10 of 18, so deciding early gives up little and saves ~1,300 tokens;
  - the 4 finished traces that narrow keep their 2 correct answers.

## What I changed and why
- **One repurposed neuron, L21 n6823** (`finalist_commit_L21`):
  - gate (w1 row): a two-finalist direction at the L21 MLP input. It is the mean difference between paragraph-break tokens ending a two-finalist paragraph and all other breaks (leave-one-trace-out AUC 0.86). The threshold uses the mean-input direction as a bias proxy;
  - up (w3 row): the bias proxy (about 1);
  - output (w2 column): commit openers (So, Thus, Therefore, Hence) minus deliberation openers (Option, (A, -, Let, But, We, ...). It is clipped to 0.98 × the Q8_0 block absmax of the other 31 entries per row.
- Paragraph-break tokens are any token containing "\n\n" (".\n\n", ":\n\n" ... 296 token types), not only token 8.

## Evidence
| set | parent → edited | diff | +/− | cut |
|---|---|---|---|---|
| DEV32 (bank: 22 narrowing + 10 other) | 7 → 10 | +9.4 (SE 5.2) | +3/−0 | 23 → 19 |
| HOLDOUT32 (fresh) | 13 → 15 | +6.25 | +2/−0 | 17 → 13 |
| HOLDOUT32-B (fresh) | – | 0.0 (SE 4.5) | +1/−1 | 9 → 9 |
| holdouts combined (64 fresh) | | **+3.1 (SE ≈ 3.1)** | +3/−1 | |

- **Read pass (teacher-forced):** where the gate fires, the deliberation opener collapses ("Option" 0.80 → 0.04, "Let" 0.61 → 0.13). Mid-list breaks are untouched, as are breaks in unrelated finished traces (0.822 → 0.823). Elsewhere the mean |Δlogp| is 0.004 per token.
- **In the generations:** after two-finalist paragraphs the edited model opens with a commit word 11 of 37 times, against 5 of 59 for the parent.
  - All 5 DEV and HOLDOUT32 gains are traces where the parent narrowed to two finalists and ran to the cap; they now finish with the right finalist.
  - The HOLDOUT32-B gain and loss do not involve the gate. The traces diverged early (resampling) and took different paths.
- **The effect depends on how many cut-and-narrowing questions a set contains:** 18 of 96 bank questions, 4 or 5 of 32 in the fresh sets. Expect a gain of about +2 to +4 pp on random questions, which is hard to resolve with 128.

## What I would try next
1. **Fire on more of the narrowing breaks.** The gate catches about 66% of two-finalist breaks at this threshold; a direction fit on more traces, or a second layer, would raise that. Hedged single-answer statements ("Thus likely answer is (E)") are a related state worth adding.
2. **Make the commit carry through.** After "Thus" the model sometimes writes "Thus we need to decide between C and H". A second gated neuron, firing on "Thus/So" right after a two-finalist paragraph and writing "the answer is" tokens, would complete the decision.
3. **Reduce resampling noise.** Every repurposed neuron diverges traces early. Use a neuron with near-zero original output, if one can be found with a cheap `mlp` read pass at `positions: "mean"`, or test with larger sets.
4. **The never-concluding knowledge questions without two finalists** (most law/health cuts) still have no lever: the model lacks the fact. Commit-type edits will not rescue them.
