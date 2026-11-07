# Generation 7 report

- Created: 2026-11-07 15:44 UTC
- Parent: `gen005_carry_gate_wrapup_L20`
- Editors: claude, ed_a; budget 6 passes each; up to 1 candidates each
- Hidden selection subset: 128 questions (seed `selection-gen-7`)

## Suggested starting points

- **claude**: Build an edit that is exactly silent where it does not fire, then use it to carry the best-evidenced mechanism. First find a carrier (a neuron, or another place for a gated function) that leaves untouched traces byte-identical in DEV generate passes (count identical traces and first-difference positions). Then put a targeted function on it, for example generation 6's two-finalist commit (the parent's 18 cut bank traces that narrow to two options). Without silence a +3 point mechanism cannot pass selection.
- **ed_a**: Answers that are stated inside the thinking and then re-verified until the cap: 18 of the 53 cut bank traces state an answer (10 of them correct) and still reach the cap. The parent's own wrap-up neuron (L20 n5414) handles some such moments; find which ones it misses and why, and repair them. Make the edit exactly silent where it does not fire (see WHAT IT TAKES TO PASS SELECTION): count byte-identical traces in your DEV passes.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 4 | read 4 texts | 104.2 |
| claude | 2 | read | original | 32 | read 32 texts | 2076.7 |
| claude | 3 | generate | edited | None | read None texts | 2798.4 |
| claude | 4 | generate | edited | None | read None texts | 1821.9 |
| ed_a | 1 | read | original | 15 | read 15 texts | 414.5 |
| ed_a | 2 | read | original | 52 | read 52 texts | 1116.6 |
| ed_a | 3 | read | edited | 82 | read 82 texts | 2585.6 |
| ed_a | 4 | generate | both | None | read None texts | 1780.0 |
| ed_a | 5 | generate | both | None | read None texts | 3029.6 |
| ed_a | 6 | generate | both | None | read None texts | 2884.5 |

## Candidates

### ed_a/ed_a_g7_short_answer_dupfold_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 20455

# ed_a_g7_short_answer_dupfold_L23: a short visible answer, hosted in freed DUPLICATE neurons of layer 23

TARGETED BEHAVIOUR. After </think> the parent re-derives the whole solution in the visible answer (~250 tokens) before
"The answer is (X)"; in 41 of 43 finished bank traces that letter equals the last letter stated inside the thinking.
When the thinking ends late (>~2,800 tokens) this redundant text is what the 3,072-token cap cuts: 5 of 53 cut bank
traces had closed the thinking, 4 of them with the right answer already stated.

CHANGE (layer 23 MLP only; w1 rows / w3 rows / w2 columns of 3 neurons + one folded w2 column).
1. Free three slots without deleting function: L23 has a cluster of 4 duplicate neurons {1228, 3401, 3793, 6986}
   (cos of w1 rows .9998, w3 rows +-.9999, w2 columns +-.9998; activation correlation .9999). Their contributions are
   algebraically merged into 1228: w2[:,1228] += w2[:,3401] - w2[:,3793] - w2[:,6986] (signs from the duplicate
   structure; no fitting). Q8 simulation: off-target change 2.6e-4 vs 1.0e-3 for removing the 3 quietest neurons.
2. Three gated units, all reading the L23 MLP input:
   - n3401 fires at </think> and writes '"The';
   - n3793 fires at '"The' right after </think> and writes ' answer';
   - n6986 fires at ')".' ending that answer line and writes <|im_end|>.
   Gate (w1 row) = shrinkage-LDA direction (lam .1) target-vs-all-other-tokens on 128k recorded tokens of 61 traces
   minus T x a whitened bias proxy (CV .03), scaled so the gate is -20 at the max negative token and +20 at the
   min positive (2-fold CV by text: pos min z 19-34, neg max z 9-15). silu(-20) quantises to exactly 0 in Q8.
   Up (w3 row) = 4 x bias proxy. Write (w2 column) = gf*(E[target] - mean E[top-10 competitors])/gpf, scaled into
   the Q8 block absmax of every row (block scales preserved exactly; cos .99 to the ideal direction).
   The answer letter is copied from the thinking's last line by the model itself (P = 1.00).

EVIDENCE.
- Read pass (46 synthetic post-think answers): P('"The'|</think>) = P(' answer') = P(<|im_end|>) = 1.000 (min);
  thinking tokens mean |dlogp| 2.7e-4 (fp32 engine), no firing on 30 unseen cut traces (~100k tokens).
- DEV (32 bank questions incl. the 5 closed-but-cut targets): 11 -> 16 correct, +15.6 pts (+6/-1); the 4 targets with
  a stated answer (1740, 6014, 8198, 9274) now finish with '"The answer is (X)"'; tokens -273.
- HOLDOUT (64 fresh practice questions, seeds 777/778): 27 -> 25, -3.1 pts (SE ~3.8; +2/-4). Neither holdout sample
  contained a closed-but-cut parent trace, so all 6 swaps are resampling (traces diverged early: only 10/64 thinking
  traces byte-identical, first difference at median 16-19% of the trace). 32 of 33 edited finished answers are exactly
  '"The answer is (X)"' with the parent's letter (one thinking ended in \boxed{F}; unit did not fire, long answer).
- Expected effect on random questions is small (closed-but-cut traces: 5 of 160 questions seen); the edit does NOT
  achieve off-target silence, so selection will mostly measure resampling noise.

## Selection (hidden questions) and decision

Parent accuracy on the 128 selection questions: **46.09%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| ed_a/ed_a_g7_short_answer_dupfold_L23 | 49.22 | +3.12 | 2.9 | +1.08 | 9 | 5 | +0.0000 | 20455 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

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

### ed_a

_(no report yet)_

