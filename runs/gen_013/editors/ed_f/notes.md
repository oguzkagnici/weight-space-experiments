# Notes of editor ed_f, generation 13


## Free diagnosis (no passes; code/001-006) -- step-level errors in the parent's 8,192 traces
- Material: 288 parent traces (passk 64x4 + bank sample 0), 15 mixed questions, 17 finished-wrong samples on them.
- ARITHMETIC SLIPS ARE NOT THE PROBLEM (larger-set check, code/003, 005): 2,021 binary statements "a op b = c" auto-verified.
  12% deviate, but 2/3 of those by < 0.3% (rounding of long decimals) and almost all larger ones are regex artefacts (chained expressions,
  sqrt(3)/2 etc.). Hand check of the 84 "large" ones: ~5 real slips (1306.25-125=1171.25, 1275-125=1100, 1125+135=1385, 15.30+1.7136=16.9136,
  1/0.075=8.333) = 0.25% of statements. Slip share is the same in right traces (4.6%) and cut traces (3.9%).
- WHERE THE FAILURES ARE: finished-wrong answers are verbal (law 21, health 12, economics 11, other 10, psychology 9, history 4 of 77);
  quantitative categories (business, chemistry, physics, math, engineering) have 5 finished-wrong but 27 of the 49 cuts. Only 7 finished-wrong
  traces contain any equation, 1 of them a slip.
- First wrong step, read by hand in wrong samples of mixed questions (vs a right sample of the same question):
  9251 (lens, B vs I): sign convention misapplied (Cartesian u=-50 put into 1/f=1/v+1/u), then "double-checked" with the same formula;
  7456 (MPC, A vs F): wrong model of the problem (consumption out of income+loan); 6110 (B vs F): unsupported inference "assigned to dose
  cohorts (randomization)", decisive cue "patients are informed" under-weighted; 2654 (G vs H): imports features not in the question (paralysis);
  5325 x2 (E vs I), 2055 x2 (C,E vs G): no knowledge, coin-flip between plausible finalists ("Hard to say", "safe answer").
  -> classes: judgement/knowledge 4, unsupported inference or cue under-weighted 2, formula/model misapplied 2; arithmetic 0, copy 0, negation 0.
- Negation questions (NOT/EXCEPT/LEAST...) are 2.2% of the practice set: too rare to matter.
- Always-cut quantitative questions (345, 774, 4152, 11720): the computed value does not match an option exactly (345: 73.15 vs gold 73.40), the
  model re-interprets the problem until the cap. Not a slip either.

## PLAN (written before any pass)
The suggested class (arithmetic/copy slips) does not hold on the larger set, so an edit that repairs digit production has no target.
What remains testable about the reasoning itself is SELF-CORRECTION (TASK, kind 2): does the network register that a step it just wrote is
wrong, and does that reach re-checking? In 9251/7456 the wrong samples "double-check" by repeating the same step, so the open question is whether
an error signal exists at all, and for which kind of error.
- Pass 1 (read, parent): teacher-forced prefixes of real traces ending in a step, each in a clean and a corrupted version, text ended with a
  paragraph break so the next token is a paragraph opener. Families: F1 arithmetic result (big: leading digit; small: late digit, exact results
  only); F2 number mis-copied from the question in the restatement. Read at the last token: top-50 (P of re-check openers Wait/Actually/Hmm/
  But/Hold...), residual h0-h24, MLP inputs of w2 L18-L23; token log-probs (P of the clean vs corrupted number).
  Decision: (a) no residual signal (held-out-by-question AUC < .7) -> the network does not register such errors; report, stop this line.
  (b) signal but weak behaviour (P(re-check) rises little) -> build a gated unit in L22/L23 (free duplicate slots n331/n5105/n881 of L23) reading
  the error direction and writing re-check openers; that is "strengthening a real error signal -> re-check pathway", not a global knob.
- Pass 2 (read, edited): GATE on held-out questions: P(re-check opener) after corrupted vs clean steps, off-target |dlogp| on clean traces.
- Pass 3 (+4): generate "edited" n=4, cap 8192 on passk questions (parent cached): conclusions wrong->right / right->wrong on both-finish.
- Last 2 passes: holdout48 at 8,192 (24 + 24).
Risk known in advance: natural consequential slips are rare (0.25% of statements), so even a working detector can only matter on few questions;
more re-checking costs tokens (gen 5: cutting re-checks is what helped). I will report that honestly.

## Pass 1 (read, parent, no edit): work/req_p1.json = 306 texts (code/007): F1 80 exact arithmetic statements from 20 questions x {clean, big (leading digit changed), small (last digit changed)}; F2 33 question restatements x {clean, number changed}. All end in a paragraph break. Returns top-50, h0-h24, MLP L18-23 at the last token, token log-probs.

## Pass 1 result (code/008-012): the parent KNOWS the right step but barely REGISTERS a wrong one once it is on the page
- Knowledge at the step: log P(result tokens) clean median 0.00 (81% of exact results have P > .5); corrupted results log P -24 (big) / -19 (small).
  Numbers copied from the question: 990 copies in the clean prefixes, median P 1.000, mean .966 (the low ones are choices of what to write next,
  not mis-copies). So neither digit production nor copying is a weak point (agrees with the 0.25% slip rate).
- Behaviour after the corrupted step (next paragraph opener, top-50): P(re-check opener: Wait/Actually/Hmm/Hold/But/However...)
  F1 clean .198 -> big .302 (+.105, SE .017) -> small .238 (+.040, SE .010); F2 clean .016 -> .149 (+.133, SE .046).
  i.e. after a result it would itself give probability e^-20, the model walks on in ~70% (F1) / 85% (F2) of cases.
- Residual at that position: the lens logit difference (re-check minus continue openers) rises by +1.55 logits (SE .14) within pairs, built up
  gradually L14 -> L24 (+.20 h14, +.59 h18, +1.01 h20, +1.44 h22, +1.65 h24), not by a few MLP neurons (L21 +.23, L22 -.03, L23 +.05 by DLA).
  But across contexts it is weak: held-out-by-question AUC of the mean-difference direction .63 (F1 big), .58 (small), .74-.79 (F2);
  shrinkage LDA at best .71 / .69 / .78 (h24); best single neuron of L18-L23 .65. Paired win share .90-.95: the direction is real, the
  context-to-context spread along it is larger than the error effect.
- Decision by the pre-registered rule: borderline (a). There is an error signal -> re-check pathway (in scope to strengthen), but no detector
  sharp enough for a threshold gate, and the target error is rare in natural traces. I still build the one edit this evidence supports, as the
  test of the pathway, and measure it: a gated L23 unit reading the LDA error direction at the MLP input and writing re-check openers.

## Pass 2 (read, parent, no edit): work/req_p2.json = the 226 pass-1 clean/big texts + 70 paragraph-break controls + 55 random-token controls + 14 prompt-token controls; records the INPUT of L23/L22 feed_forward, mlp23, h22-24, top-50 at the last token (needed to place a gate at the L23 MLP).

## Pass 2 result (code/014, 015): a gate at the L23 MLP input
- Recorded input reproduces mlp23 exactly (corr 1.0000). Cross-validated (6 folds by question) shrinkage-LDA detector, lam 3:
  AUC F1 .705, F2 .793, corrupted vs generic controls .83; at 5% false positives overall: fires on 29% of corrupted F1 steps, 27% of F2,
  10% of clean arithmetic steps, 1% of generic paragraph-break / random-token controls. Same numbers at L22 input, h22, h24.
## EDIT v1 (staged in edited/; code/014 `0.03 3.0 save`; params in work/edit_params.json)
- model.layers.23.feed_forward.{w1,w2,w3}: neuron 881 freed by folding into its near-duplicate (w2[:,331] += w2[:,881]; cos .9996).
- w1[881] = kap*(P w - theta*m_hat/mp): w = unit shrinkage-LDA direction (lam 3) of corrupted-minus-clean at the L23 MLP input, fitted on HALF of the
  pass-1 questions (work/train_questions.json: 57 pairs F1-big + F2) with clean + half of the controls as within-class scatter; P projects off the
  inputs of the first 8 prompt tokens; theta 1.41 = 97th percentile of out-of-fold negative scores; m_hat/mp = mean-input bias proxy; kap = 4/sd.
- w3[881] = m_hat/mp (about 1). w2[:,881] = s*(g_final*r)/g_post, r = mean unembedding of {Wait, Actually, Hmm, Hold, But, However} minus
  {So, Thus, Then, Now, Therefore, Hence, Next}; s set for +3 logits (re-check minus continue) at the median firing activation; column norm .37,
  nothing clipped by the Q8 block rule.
- Predicted on the held-out half (recorded inputs): shift after corrupted F1 +1.52 logits vs clean +0.43; F2 +2.58 vs 0.00; controls +0.01.
## Pass 3 (read, edited = v1): the 306 pass-1 texts, top-50 at the last token + token log-probs -> GATE on held-out questions:
   P(re-check opener) after corrupted vs clean steps (big, small, F2), and off-target |dlogp| over all 184k tokens vs pass 1.

## Pass 3 result = GATE for v1 (code/016; edited vs pass 1 on the same 306 texts)
- HELD-OUT questions (not used to fit the gate): P(re-check opener) F1 clean .217 -> .242 (+.025, SE .013); F1 big .327 -> .396 (+.069, SE .025);
  F1 small (never used anywhere) .266 -> .359 (+.093, SE .031); F2 clean .002 -> .002; F2 big .129 -> .317 (+.187, SE .071).
  Corrupted-minus-clean gap: big .110 -> .154 (+.045, SE .020), small .049 -> .117 (+.068, SE .025), F2 .127 -> .314 (+.187, SE .071).
  Top-1 opener changed in 7/41, 7/41, 4/15 corrupted texts vs 2/41, 0/15 clean. -> the edit strengthens the error -> re-check response
  specifically (2-3x the existing gap), on held-out texts. Fit questions: smaller (+.017 / +.007 / +.075).
- Off-target (184k teacher-forced tokens): mean |dlogp| .0038, median .00006, 2.0% of tokens > .01, 0.55% > .1. The 131 distinct positions with
  |d| > .3 are paragraph/sentence openers after a doubtful statement ending in "?" (raises Actually/Wait, lowers Let/Or/Usually): the direction
  also reads generic doubt, not only numeric errors.
## Passes 4-6 (generate, edited = v1, cap 8192), queued together (one edit only, to save queue time):
- Pass 4 DEV: work/req_dev.json, 6 passk questions x 4 samples (parent cached): quantitative mixed 9251 (BIII), 7456 (AF-C), 11805 (BJJD),
  7261 (-DDD); verbal mixed 6110 (BBBF); always-right control with many equations 8198 (CCCC).
- Passes 5-6 HOLDOUT48: work/req_hold1.json (ids 0-23), work/req_hold2.json (ids 24-47).

## Pass 4 = HOLDOUT48 second half (ids 24-47), edited v1 (the first submissions of dev/hold1 failed on a full disk, not charged; re-queued as passes 5 (DEV) and 6 (hold1))

## Pass 4 result (HOLDOUT48 ids 24-47, v1; code/017, 018)
- overall 62.5 -> 50.0 (-12.5, SE 6.9, +0/-3); both-finish (19) 63.2 -> 63.2 (+0/-0; 4 answers changed, all wrong->other wrong);
  cuts 2 -> 5: all three losses are finished-right -> cut. Re-check openers (Wait/Actually/Hmm/However at sentence starts) per 1,000 tokens
  4.35 -> 6.74 (+55%), per trace 15.7 -> 26.5. First difference at a median 5.8% of the trace.
- Reading: v1 is too broad. Its direction also reads generic doubt (statements ending in "?"), so in generation it acts as a re-check booster in
  verbal traces, which costs finished answers (the opposite of what generation 5 gained). The three losses are 2 "other" + 1 chemistry.
## EDIT v2 (staged; code/014 `0.01 3.0 save`): same unit, same direction, threshold raised to the 99th percentile of out-of-fold negatives
  (theta 1.41 -> 1.70). Predicted from recorded inputs on held-out questions: fires on 24% of corrupted F1 steps (v1 32%), 10% of clean ones (15%),
  0% of paragraph-break controls (3%), 4% of random-token controls (8%); mean shift corrupted F1 +0.61 logits (v1 +1.52), clean +0.11 (+0.43),
  F2 +1.15 vs 0, controls -0.002 (v1 +0.013).
## Pass 5 = DEV (v1, already running when v2 was staged). Passes 6-8 (edited = v2): 6 = gate read (the 306 pass-1 texts, as pass 3);
   7-8 = HOLDOUT48 (ids 0-23, 24-47), queued in one chain to fit the remaining budget; v1's holdout stays at the 24 questions of pass 4.

## Pass 5 result (DEV, v1; 6 passk questions x 4 at 8,192, parent cached)
- overall 58.3 -> 75.0 (+16.7, SE 10.5 clustered by question, +5/-1); both-finish (20) 70.0 -> 80.0 (+9.7, SE 10.6; +3/-1, 6 answers changed);
  cuts 2 -> 2 (2 cut -> finished right, 2 finished-wrong -> cut); tokens 3,966 -> 4,380; re-check openers per 1,000 tokens 3.39 -> 5.92.
  9251 BIII -> IIII, 7456 AF-C -> FFFA, 11805 BJJD -> HDJ-, 7261 -DDD -> DDDD, 6110 BBBF -> F-BF, control 8198 CCCC -> CCCC.
- Caveat: 5 of the 6 questions were chosen because the parent has wrong samples on them; every edited trace differs from the parent's within
  the first 0-17% (median 6.7%), so wrong samples of mixed questions are simply re-drawn (regression to the mean). Not evidence by itself.
  Wrong -> right on both-finish: 9251 s0, 7456 s0, 6110 s0; right -> wrong: 11805 s1.
## Pass 6 result = GATE for v2 (edited vs pass 1, same 306 texts)
- HELD-OUT questions: P(re-check opener) F1 clean .217 -> .226 (+.009, SE .008); big .327 -> .368 (+.041, SE .018); small .266 -> .310
  (+.045, SE .018); F2 clean .002 -> .002; F2 big .129 -> .218 (+.088, SE .038). Corrupted-minus-clean gap: big +.032 (SE .016), small +.036
  (SE .016), F2 +.088 (SE .038). Top-1 opener changed in 6/41, 1/41, 1/15 corrupted texts, 0 clean.
- Off-target: mean |dlogp| .0020 (v1 .0038), 1.1% of tokens > .01 (2.0%), 0.32% > .1 (0.55%).

## FINAL (passes 7-8, edit v2; code/017, 018)
- HOLDOUT48 at 8,192, v2: overall 62.5 -> 62.5 (+0.00, SE 2.98, +1/-1); both-finish (40) 75.0 -> 75.0 (+0.00, SE 3.58, +1/-1), 7 of 40 answers
  changed; cuts 7 -> 7; tokens 3,472 -> 3,270; re-check openers per 1,000 tokens 4.09 -> 5.20; 4/48 traces identical, first diff at 8.8%.
- Submitted: ed_f_g13_error_recheck_gate_L23 (= v2; 3 tensors, 8,188 weights). Report in report.md. All 8 passes used.
