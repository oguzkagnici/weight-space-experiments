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
