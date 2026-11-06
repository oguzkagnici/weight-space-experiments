# gate_wrapup_L20: a gated "stop re-checking" neuron in layer 20

Edit: repurpose the quiet MLP neuron 5414 of layer 20 (model.layers.20.feed_forward w1 row, w3 row, w2 column; about 2,048 weights each).
- gate (w1 row) = commit direction - threshold x bias proxy. The commit direction is the mean-difference direction between
  residual states at points where the model has just stated its answer inside <think> and matched control points, read from recorded activations.
  The threshold is at the 99.8th percentile of all-token scores (s > 10.28), so it fires on about 0.2% of tokens.
- up (w3 row) = bias proxy (a direction with a near-constant projection, so the neuron's output is about gate-score minus threshold).
- down (w2 column) = (wrap-up word direction - reopen word direction) mapped back through the final norm (regularised inverse),
  scaled so that the estimated logit-difference shift at commit points is -9 (scale 0.400).
  Wrap-up words exclude </think> itself.
No training, no fitting beyond means and percentiles of recorded activations.

Mechanism measured (pass 3, read): at commit points that reopened, P(reopen) 0.81 -> 0.68 and P(close) 0.11 -> 0.24 (at target -3).
Control tokens are unchanged; the neuron fires (> 0.5) on 71% of commit points and 10% of controls.

Evidence (thinking mode, 3072-token cap, fixed seeds, paired original vs edited):
- DEV (32 practice questions, used for tuning): 37.5 -> 46.9 (+9.4, SE 6.9; +4/-1); cut share 0.53 -> 0.31; tokens -377.
- HOLDOUT A (32 fresh practice questions, untouched): 31.2 -> 34.4 (+3.1, SE 3.1; +1/-0); tokens -50.
- HOLDOUT B (32 fresh practice questions, untouched): 9.4 -> 18.8 (+9.4, SE 5.2; +3/-0); cut 0.81 -> 0.69; tokens -253.
- HOLDOUTS combined (64): +6.3 pp (SE about 3.0; +4/-0).
Where it helps: traces that state an answer and then re-check until the cap; it does nothing for traces that never commit.
Risk: on finished traces it shortens the thinking (same answers so far); one DEV loss (mmlupro_9424).
