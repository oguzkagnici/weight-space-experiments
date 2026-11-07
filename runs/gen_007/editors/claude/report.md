# Report of editor `claude`, generation 7 (no candidate submitted)

## The brief
Build an edit that is exactly silent where it does not fire, so that untouched traces stay byte-identical and a
targeted mechanism (gen 6's two-finalist commit) can pass selection.

## Result: exact silence is not reachable under the rules
- **No repurposable MLP neuron is silent after Q8_0 quantisation.** The quietest neuron per layer (L12-L23) still has a nonzero Q8 activation on 30-80% of tokens; the median neuron on ~95%. No w2 column quantises to all zeros (each keeps at least 1,934 of 2,048 nonzero entries). Sources: pass 1, code/003 and code/005.
- **Removing even a tiny original output resamples every trace.**
  - Edit v1 used L22 n1786, the second-lowest removal impact of all 184k neurons in L12-23.
  - Its new gate is silent before divergence: |h_new|·||col|| = 0.000 before the first diverging token in 31 of 32 traces (code/012).
  - Yet 95 of 96 traces diverge from the parent, median at 5.6% of the trace. The cause is the removed original output, about 0.02 per token. At temperature 0.6, a perturbation that small flips a sampled token within a few hundred tokens.
  - The gen-6 calibration of the removal-impact score (s 0.0013 predicting ~22-25%) does not transfer across layers.
- **The one exactly silent carrier I found is out of scope.**
  - It is the tied embedding row of a token that never occurs in the context.
  - Sampling applies top-k 20 first, so that token's logit is irrelevant wherever it stays outside the top 20. Its input embedding is never read.
  - A row there could implement a linear gate that is exactly silent off-target, but `model.embed_tokens` may not be edited.
- **A length-triggered stop would also need no silence**, because the cap is the problem. It is out of scope (earlier decision).

## Edit v1 (not submitted): the two-finalist commit gate moved to L22 n1786
- gate (w1 row): the gen-6 mean-difference direction between paragraph breaks ending a two-finalist statement (38) and other breaks (1,504). It is refit at the L22 MLP input (leave-one-trace-out AUC 0.864) with the bias-proxy threshold (thr 7.5, k 6).
- **New detail.** The bias-proxy gate fired on the first prompt tokens of every text: 'user' at L22; BOS, `<|im_start|>` and `\n` at L23. These are constant, low-norm attention-sink tokens. Firing there would change every question through the KV cache.
  - The fix projects the gate row off their 4-dim subspace and subtracts 0.5 × the sink direction. Afterwards the prompt does not fire.
  - Coverage: 79% of two-finalist breaks, 13.7% of other breaks and 0.27% of all tokens. (The gen-6 L21 gate may have had the same sink issue; it was not checked.)
- up (w3 row) = mu_hat/mp; down (w2 column) = commit openers minus deliberation openers, clipped per row to the Q8 block absmax (norm 1.47).

## Evidence (generate, edited only, paired against the trace bank)
The bank reproduces the parent byte-identically: 32/32 gen-6 originals matched. So the 96 bank questions give a 96-question paired test at the cost of 96 generations.

| set | parent → edited | diff | +/− | cut | byte-identical |
|---|---|---|---|---|---|
| DEV32 (18 cut-narrowing, 4 finished-narrowing, 10 other) | 8 → 6 | −6.2 (SE 6.2) | +1/−3 | 21 → 21 | 0/32 |
| EXTRA16 | 10 → 10 | 0 | 0/0 | 6 → 6 | 0/16 |
| REST48 | 19 → 22 | +6.2 (SE 3.5) | +3/−0 | 26 → 23 | 1/48 |
| **all 96** | 37 → 38 | **+1.0 (SE 2.8)** | +4/−3 | 53 → 50 | 1/96 |

- **Mechanism:** after a two-finalist paragraph the edited model opens with a commit word 10/29 times, against 3/39 for the parent (gen 6: 11/37 vs 5/59). So the gate does what it was built for.
- **Accuracy:**
  - Of the 18 cut-narrowing DEV traces, 4 now finish but only 1 is correct.
  - The REST48 gains are in "other" traces, which the gate does not target: resampling.
  - Correctness changes on 7 of 96 questions and cut status on 13, almost all by resampling. That is the noise floor any candidate faces: SE ≈ 2.4 pp at 128 questions.
- Expected selection outcome for v1: z ≈ 0.3. Not submitted, to save the selection compute.

## Recommendations
1. **Size the mechanism, not the silence.** Under these rules every edit resamples. A candidate needs about +5 pp or more on random questions to pass z ≥ 2 at 128 with useful probability.
   - Two-finalist oscillation is 18 of 53 cut bank traces; even a perfect commit there is worth about +4 pp on the bank and less on random questions.
   - Mechanisms worth stacking must together cover most cut traces: two-finalist commit, answer stated then re-verified (ed_a's area), and hedged statements.
2. **Use the trace bank for paired testing.** Edited-only generate on bank questions is a free paired comparison (48 per pass); 96 paired questions cost 2 passes.
3. **Rule question for the experimenters.** If exact silence is the goal, the tied embedding row of an unused token is the natural carrier. It is forbidden by the paper's scope; changing that is a decision for the experiment, not for an editor.
4. **Check the bias-proxy gate on prompt-sink tokens** in every gated edit, including the parent's L20 wrap-up neuron.
