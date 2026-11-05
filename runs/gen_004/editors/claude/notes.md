# Notes of editor claude, generation 4

## Diagnosis on the trace bank (free)
- 60 of 96 bank traces state an answer inside the thinking ("answer is (X)"); all 43 finished ones do, 17 of 53 cut ones do.
- Finished traces: the answer changes after the first statement in only 2 of 43 (one wrong->right, one wrong->wrong); median 347 tokens after the first statement (incl. ~245 tokens of the post-</think> answer).
- Cut traces with a statement: first stated answer correct in 9 of 17, lost to the cap. Upper bound of "stop re-checking after a commitment": about +8 of 96.
- After a line stating an answer, the next line starts with a reopening word (But/Let/However/Check/Wait) in 31 of 58 cases, a wrap-up word (We/Thus/Hence/Now) in 20.
- Hypothesis H1: a component writes "reopen / double-check" after a committed answer; damping it specifically at commitment points (not at ordinary paragraph ends) saves the post-commit re-verification budget.

## Pass 1 (read, original): 58 commit points + 25 reopen + 25 close control paragraph ends (texts cut at a line end). Last position: residual 0-24, MLP 0-23, o_proj inputs of all attention layers, top-30, attn_last 10/14/18/22.
### Pass 1 results (read, original, 108 decision points)
- Parent's next-token mass at commit points: reopen 0.81 / close 0.11 where it reopened (n 31); 0.32 / 0.61 where it wrapped up (n 20). Controls (paragraph ends): 0.71/0.09 (reopen), 0.04/0.71 (close). The parent predicts its own continuation well.
- DLA (logit diff reopen-close, through post_ffn_norm/post_operator_norm and the final norm): no component pushes reopen specifically at commit points; per-component commit-control differences are <= 0.14 logits (L20.H8 +0.14, L16.H15 +0.08, L19 n1851 +0.07). Reopening is distributed.
- Commit detectors: no single MLP neuron (best AUC 0.90, early-layer lexical features, small activations). Residual mean-difference AUC ~0.9 at all layers; CV AUC of MLP-activation mean-difference readouts 0.83-0.89 (separation ~1.8 sd); per-head outputs CV separation <= 3.0 sd (L22.H10), attention to the statement span differs by <= 0.03.
- Conclusion: the "answer just stated" condition is distributed, so a linear (rank-one W2) edit would leak onto ordinary paragraph ends. Plan: repurpose one quiet MLP-20 neuron as a gated detector (gate = commit direction minus threshold via the near-constant component of the normed input; up = constant component; out = wrap-up direction). Needs the MLP-20 input at all tokens to set the threshold -> pass 2.

## Pass 2 (read, original): 10 full bank traces (5 cut with a statement, 5 finished), all positions: h20, h21, mlp20, token log-probs.
