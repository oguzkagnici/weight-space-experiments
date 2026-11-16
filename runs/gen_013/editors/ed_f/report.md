# Report: editor ed_f, generation 13

**Candidate submitted:** `ed_f_g13_error_recheck_gate_L23` (3 tensors, 8,188 weights). All 8 passes used: 4 read, 4 generate.
**Holdout48 at 8,192 tokens:** overall +0.00 points (SE 2.98, +1/-1), 62.5 -> 62.5%. Both-finish (40 questions): +0.00 (SE 3.58, +1/-1), 75.0 -> 75.0%.
**Expected on selection:** about 0. GSM8K was not measured.

## What the model gets wrong: not step slips
I followed the suggested lead (first wrong step in wrong samples of mixed pass@4 questions) and checked the main class on a larger set before spending passes.

- **Arithmetic slips are rare and unrelated to failure.** I auto-verified 2,021 statements of the form "a op b = c" in 288 parent traces at 8,192 tokens.
  - 12% deviate, but two thirds by less than 0.3% (rounding of long decimals) and nearly all larger ones are parsing artefacts.
  - By hand, about 5 are real consequential slips (0.25%), for example 1306.25 - 125 = 1171.25.
  - The deviation rate is the same in right traces (4.6%) and cut traces (3.9%).
- **Copying from the question is not a weak point.** 990 numbers copied from the question into the reasoning have median probability 1.00 (mean .966); the low ones are choices of what to write next, not mis-copies.
- **Negation questions (NOT/EXCEPT/LEAST) are 2.2% of the practice set.**
- **Finished-wrong answers are verbal; quantitative failures are cuts.**

  | subjects | right | finished wrong | cut |
  |---|---|---|---|
  | business, chemistry, physics, math, engineering | 75 | 5 | 27 |
  | all others (law, health, economics, psychology, ...) | 87 | 72 | 22 |

  Only 7 finished-wrong traces contain any equation, and 1 of them a slip.
- **First wrong step in 8 wrong samples of 6 mixed questions (read by hand against a right sample):**
  - 4 are coin-flips between plausible finalists with no knowledge ("Hard to say", "a safe answer").
  - 2 are unsupported inferences that under-weight a cue in the question: "assigned to dose cohorts (randomization)" although the stem says the patients are informed; "sleep paralysis" for a plain nightmare.
  - 2 are a formula or model misapplied: a Cartesian sign put into the real-is-positive lens formula and then "double-checked" with the same formula; consumption computed out of income plus loan.
  - None is an arithmetic, copying or negation slip.
- **Always-cut quantitative questions are not slips either.** The computed value does not match an option exactly (73.15 against a gold of 73.40) and the model re-interprets the problem until the cap.

## Self-correction: the network knows the right step but barely registers a wrong one
Pass 1 teacher-forced 306 prefixes of real traces, each ending in a step in a clean and a corrupted version, followed by a paragraph break.

- **It knows the step.** Exact arithmetic results have median log-probability 0.00; corrupted ones -24 (leading digit changed) and -19 (last digit changed).
- **The behavioural response is weak.** Probability that the next paragraph opens with a re-check word (Wait, Actually, Hmm, But, However ...):

  | step | clean | corrupted | difference (SE) |
  |---|---|---|---|
  | arithmetic result, leading digit (80 pairs) | .198 | .302 | +.105 (.017) |
  | arithmetic result, last digit (80 pairs) | .198 | .238 | +.040 (.010) |
  | question number mis-copied in the restatement (33 pairs) | .016 | .149 | +.133 (.046) |

  After a result it would itself give probability e^-20, the model walks on in about 70% of cases.
- **The residual signal is real but small against context.** The lens logit difference (re-check minus continue openers) rises +1.55 (SE .14) within pairs, built up gradually from L14 to L24, not by a few neurons. Across contexts it separates poorly: held-out-by-question AUC .63-.79 for mean-difference and shrinkage-LDA directions at h14-h24 and at the L22/L23 MLP inputs; the best single neuron of L18-L23 reaches .65.

## What I changed
- **Unit:** L23 MLP neuron 881, freed by folding it into its near-duplicate 331.
- **Gate:** a shrinkage-LDA direction of corrupted-minus-clean inputs of the L23 MLP (recorded in pass 2, fitted on half of the questions), minus a threshold, projected off the first prompt tokens.
- **Write:** re-check openers minus continue openers, +3 logits at the median firing activation.
- **Two thresholds were tried.** v1 used the 97th percentile of out-of-fold negatives; v2 (submitted) the 99th.

## Evidence
- **Read gate on questions not used to fit (edited vs parent, same texts):**

  | version | clean arithmetic step | leading-digit error | last-digit error | mis-copied number | off-target mean abs. change in log-prob |
  |---|---|---|---|---|---|
  | v1 | +.025 (SE .013) | +.069 (.025) | +.093 (.031) | +.187 (.071) | .0038 |
  | v2 (submitted) | +.009 (.008) | +.041 (.018) | +.045 (.018) | +.088 (.038) | .0020 |

  Both raise the re-check response after corrupted steps more than after clean ones. v1's off-target changes sit at openers after doubtful statements ending in "?": the direction also reads generic doubt.
- **v1 in generation behaves as a general re-check booster.**
  - Holdout48 ids 24-47: overall -12.5 (SE 6.9, +0/-3; three finished-right answers became cuts), both-finish (19) +0.0 (+0/-0), re-check openers per 1,000 tokens 4.35 -> 6.74.
  - Pass@4 dev (6 questions x 4 samples, cap 8,192): overall +16.7 (SE 10.5, +5/-1); both-finish (20) wrong->right 3, right->wrong 1. This is not evidence for the mechanism: 5 of the 6 questions were chosen for parent failures, and every edited trace diverges within the first 0-17% (median 7%), not at an erroneous step, so wrong samples are simply re-drawn.
- **v2 on holdout48 (all 48):** overall +0.00 (SE 2.98, +1/-1); both-finish (40) +0.00 (SE 3.58, +1/-1), 7 answers changed; cuts 7 -> 7; tokens 3,472 -> 3,270; re-check openers per 1,000 tokens 4.09 -> 5.20. The pass@4 generate was not repeated for v2 (no passes left).
- **Why it cannot gain:** the errors the unit detects almost never occur naturally, and the wrong conclusions come from judgement and modelling errors that the model re-confirms when it re-checks.

## What I would try next
1. **Stop looking for local slips.** Arithmetic, copying and negation are measured and clean; a detector for them has nothing to fire on.
2. **Test whether a re-check can change a wrong model of the problem.** In the lens and consumption examples the re-check repeats the same assumption. Teacher-force wrong samples up to their check paragraph and measure the probability of an alternative-formulation opener ("Alternatively, using ...") against a repeat; if a different derivation is never proposed, re-checking cannot help and that line can be closed.
3. **Cue under-weighting is the one verbal class with a possible mechanism.** Measure attention from late reasoning positions to the question stem in right and wrong samples of the same question (`attn_last` on prefixes), on far more than the 2 cases found here, before any edit.
4. **Do not use dev sets picked from parent failures.** My dev pass showed +5/-1 for an edit that lost 3 answers on holdout; any early-diverging edit re-draws wrong samples of mixed questions.
