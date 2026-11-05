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
### Pass 2 results (read, original, 10 full traces, all positions)
- MLP-20 input x-hat reconstructed from recorded h21 and mlp20 (h21 - post_ffn_norm(W2 a), then ffn_norm): recomputed neuron activations match the recorded ones (corr 0.99999).
- Commit score s = v.x-hat (v = mean commit - mean control x-hat at the decision points, orthogonalised to the mean x-hat): commit points mean 13.4, controls -2.0, all 24,845 tokens mean 0, all-token percentiles 99 / 99.8 / 99.9 = 3.7 / 10.3 / 12.5. Out of sample (2-fold), the 99.9-pct threshold passes 66% of commit points and 8% of controls.

### Current edit (candidate "gate_wrapup_L20", pass 3 = first test)
- Layer 20 MLP neuron 5414 (the quietest by max |a|*|w2 col| over all recorded tokens; max contribution 0.087 vs median neuron 0.20) repurposed:
  - w1 row (gate) = v - th * mu_hat / m (th = 10.28, the all-token 99.8th percentile of s; mu_hat / m = normalised mean x-hat, ~1 on typical tokens);
  - w3 row (up) = mu_hat / m;
  - w2 column = 0.083 * w, w = regularised (gpost / (gpost^2 + 0.02^2)) map of the residual direction -(gF * (mean E[reopen words] - mean E[wrap-up words])); wrap-up words: Thus, Hence, Therefore, So, We, Final, Answer; NOT </think>.
- Pre-screen (linear read-out through post_ffn_norm and the final norm, ignoring layers 21-23): commit points median -3.0 logits on (reopen - close) (25% of them untouched), controls median 0.0 (8% below -1), all tokens: 0.12% with |change| > 1.

## Pass 3 (read, edited): 108 decision texts (last position: top-30, mlp20) + 5 finished full traces (token log-probs vs pass 2).
### Pass 3 results (read, edited at target -3; in-sample decision points)
- P(reopen) / P(close) at the next token, original -> edited:
  - commit points that reopened (n 31): 0.812 -> 0.684 / 0.112 -> 0.236
  - commit points that wrapped up (n 20): 0.320 -> 0.155 / 0.610 -> 0.783
  - controls (paragraph ends) that reopened (n 25): 0.707 -> 0.692 / 0.087 -> 0.101
  - controls that wrapped up (n 25): 0.042 -> 0.042 / 0.711 -> 0.711
  - P(</think>) stays 0.000 everywhere
- New neuron activation: commit points median 7.2 (71% > 0.5); controls median 0.0 (10% > 0.5).
- Side effects on 5 finished full traces (8,076 tokens, token log-probs vs pass 2): mean change < 0.001 per token; one position with |change| > 0.5, and it is a commit point ("Hence the answer is (D).\n\n" -> "But": -0.71).
- The real shift (~ -1 logit odds) is about 1/3 of the read-out estimate (-3): layers 21-23 dilute it.
- Dose for pass 4: estimate target -9 (w2 scale 0.40; scale 0.187 at -6; saturation beyond, at scale 50 80% of tokens are perturbed). Expected real shift ~ -3.

## Pass 4 (generate both, DEV): 17 cut bank traces with an in-think statement + 15 random finished ones (work/dev_split.json). Edit = gate_wrapup_L20 at target -9.
### Pass 4 results (generate both, DEV 32, edit at target -9)
- ALL 32: acc 0.375 -> 0.469 (paired +9.4 pts, SE 6.9; +4/-1); cut share 0.53 -> 0.31; tokens 2,356 -> 1,979 (-377, SE 113).
- 17 cut traces with an in-think statement: 0/17 -> 4/17 correct (+4/-0); cut 17 -> 9; tokens -540 (SE 175). Several now close right after the first statement (some wrong first answers: 10935, 4811, 7187, 1375).
- 15 finished: 12 -> 11 correct (+0/-1: 9424 closed on a wrong first statement, E instead of J); tokens -191 (SE 126); one became cut (6800, wrong before).
- 8 of the 17 still produce no in-think statement before the cap (unchanged by design).
- Expectation for random questions: about 18% of questions are cut-with-statement (+0.24 each) and 45% finished (-0.07 each) -> roughly +1 to +4 pts net; a 32-question HOLDOUT has SE ~5 pts, so passes 5 and 6 both go to HOLDOUT (64 questions in total) at this dose.

## Pass 5 (generate both, HOLDOUT A: 32 random practice questions not in the bank, work/holdout_ids.json). Same edit.

## Pass 6 (generate both, HOLDOUT B: 32 more random practice questions, work/holdout_b_ids.json). Same edit.

## Pass 5 result (HOLDOUT A, 32 fresh practice questions, same edit, target -9), code/016
- acc 0.312 -> 0.344 (paired +3.1, SE 3.1; +1/-0: mmlupro_2705 cut -> finished correct F). Cut share 0.62 -> 0.62. Tokens 2512 -> 2462 (-50, SE 76).
- Much weaker than DEV (+9.4). HOLDOUT A has few cut-with-statement traces (most cut traces never state an answer: 0 statements), and those are the only place the gate acts. Finished traces mostly got shorter (e.g. 5458 2964 -> 1770, 3198 1158 -> 872) with the same answer.
- One right->? check: mmlupro_6268 was finished-wrong (D) and is now cut (still wrong), so no loss.
## Pass 6 (HOLDOUT B) failed on a machine-wide full disk (GGUF scratch); not charged; requeued unchanged (same snapshot).
