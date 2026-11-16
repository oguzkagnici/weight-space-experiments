# Generation 13 report

- Created: 2026-11-15 22:57 UTC
- Parent: `gen008_carry_stack_shortans_L23_finalist_L22`
- Editors: claude, ed_f; budget 8 passes each; up to 1 candidates each
- Hidden selection subset: 256 questions (seed `selection-gen-13`)

## Suggested starting points

- **claude**: SELF-CORRECTION. Does the network notice its own wrong steps? Build matched pairs from practice traces: a clean reasoning prefix, and the same prefix with ONE step corrupted (a number, sign, unit or negation changed so the step is false). In teacher-forced read passes, measure what follows the corrupted step vs the clean one: P of re-check tokens ('Wait', 'Actually', 'Let me double-check'), and residual-stream directions that separate corrupted from clean at the step end (fit on some pairs, test on held-out pairs). Then ask whether the signal REACHES the behaviour: in the parent's own wrong traces (pass@4 bank), is the error signal present at the first wrong step but not followed by a re-check? If there is a real detector whose link to re-checking is weak, strengthen that link (only when the detector fires; raising re-checking everywhere is a global knob and out of scope) and show on held-out corrupted pairs that detection->re-check rises while clean steps are left alone. Then test generated conclusions on mixed pass@4 questions. If the network has no error signal, say so clearly: that is an important negative result.
- **ed_f**: STEP-LEVEL REASONING ERRORS. Use the pass@4 bank at 8,192 (same question, right and wrong samples). For each finished WRONG sample of a mixed question, find the FIRST false step and classify it (false fact recalled; arithmetic/algebra slip; number or condition copied wrongly from the question; NOT/EXCEPT/units/sign lost; formula or definition misapplied; unsupported leap), and note what the right samples did at the same point. Check that the most frequent class holds on a larger set before spending passes on it (your gen-12 diagnosis did not replicate at 256 questions). Pick the class that is most frequent AND has a findable cause; localise the cause with teacher-forced read passes (e.g. at the error token: which heads attend to the source in the question, which MLP neurons write the wrong value; compare with right samples at the same point), edit it, and show on held-out texts that P(correct step) rises relative to P(wrong step) before you generate. Arithmetic or copying slips are a good place to start if they are frequent, because they also matter for GSM8K.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 280 | read 280 texts | 1632.8 |
| claude | 2 | read | original | 102 | read 102 texts | 1466.0 |
| claude | 3 | read | edited | 212 | read 212 texts | 1649.9 |
| claude | 4 | read | original | 32 | read 32 texts | 266.8 |
| claude | 5 | generate | edited | None | read None texts | 2107.4 |
| claude | 6 | generate | edited | None | read None texts | 1722.1 |
| claude | 7 | generate | edited | None | read None texts | 1721.5 |
| ed_f | 1 | read | original | 306 | read 306 texts | 1538.1 |
| ed_f | 2 | read | original | 365 | read 365 texts | 1284.0 |
| ed_f | 3 | read | edited | 306 | read 306 texts | 1553.6 |
| ed_f | 4 | generate | edited | None | read None texts | 2190.4 |
| ed_f | 5 | generate | edited | None | read None texts | 1953.8 |
| ed_f | 6 | read | edited | 306 | read 306 texts | 1649.2 |
| ed_f | 7 | generate | edited | None | read None texts | 2559.2 |
| ed_f | 8 | generate | edited | None | read None texts | 1797.3 |

## Candidates

### claude/claude_g13_selfcheck_recheck_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 8191

claude_g13_selfcheck_recheck_L23 (generation 13, target SELF-CORRECTION; expected ~0, submitted as the measured test of the mechanism).
CHANGE: L23 MLP neuron 881 (folded first into its near-duplicate n331, cos .9996, w2[:,331] += .989*w2[:,881]) rebuilt as a gated unit: gate (w1 row) = 4*(w_in - T*p), w_in = corrupt-minus-clean mean direction of normalised h22 at sentence ends of 70 TRAIN corrupted/clean step pairs, divided by ffn_norm gain and projected off the prompt-sink subspace; p = bias proxy over 14.8k background positions of practice traces; T = 99.9th percentile of background scores (fires on ~0.1% of tokens). Up (w3 row) = p. Write (w2 col) = re-check openers (Wait, " Wait", Actually, " Actually") minus mean embedding, Q8-block-safe (norm 1.10). 3 tensors (L23 w1/w3/w2), 2 neurons touched.
MECHANISM / EVIDENCE: (1) Read pass 1 (140 synthetic pairs: one step corrupted by antonym/negation swap or wrong arithmetic result): the corrupted token is very surprising (logprob -13 to -21 vs -0.6 clean); P(re-check opener next) rises x1.6 (swaps); a corrupt-vs-clean direction separates HELD-OUT pairs 0.83-0.90 (paired) at L8-L24 and transfers between swaps and arithmetic, but unpaired AUC is only .63-.71. (2) Read pass 2: the signal is SELF-CONSISTENCY, NOT TRUTH: substituting GOLD for the model's own wrong pick scores MORE error-like (25/25), and the corrected step of a natural sign error (9251) scores more error-like than the model's own false step. (3) Held-out read gate (pass 3, an earlier leaky version E1 of this unit): re-check after held-out corrupted steps 0.12 -> 0.39, but also after clean steps 0.06 -> 0.23, after the gold substitution 0.88 -> 1.00 and after the corrected 9251 step 0.58 -> 1.00, while after the model's own wrong picks 0.75 -> 0.77. (4) Calibrated on real background positions (pass 4), no gate can isolate corrupted steps (0/12 held-out corrupted sentence ends fire at 0.1% background rate); E2 is that strict gate.
GENERATED: pass@4 mixed questions (6 questions x 4 samples, chosen for most finished-wrong parent samples, so regression to the mean is expected): overall +16.7 (SE 13.0, +7/-3), both-finish +22.7 (SE 13.1, n 22, W->R 7, R->W 2); 0/24 traces identical, re-check rate unchanged (1.29 -> 1.36 per 1k tokens) => resampling, not mechanism. HOLDOUT48 at 8,192: overall +0.00 (SE 4.21, +2/-2), both-finish +0.00 (SE 3.67, n 39, +1/-1); re-check rate 3.46 -> 3.45 per 1k tokens.
CONCLUSION: negative result. The network's only error signal flags deviations from its own prediction; strengthening it cannot correct confident wrong beliefs and pushes doubt onto true statements. Expect ~0 at selection.

### ed_f/ed_f_g13_error_recheck_gate_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 8188

# ed_f_g13_error_recheck_gate_L23: strengthen the network's own "a step I just wrote is wrong" -> re-check response (gated unit, L23 MLP neuron 881)

TARGETED BEHAVIOUR (self-correction). Teacher-forced on its own traces, the parent knows the right step (exact arithmetic results: median P 1.00; numbers copied from the question: median P 1.00) and gives a corrupted result log P -19 to -24, but once the wrong value is on the page it mostly walks on: P(re-check opener Wait/Actually/Hmm/But/However at the next paragraph) is .198 after a clean step, .302 after a leading-digit error, .238 after a last-digit error; .016 -> .149 after a number mis-copied from the question (pass 1, 306 texts). The residual carries the signal (+1.55 logits re-check minus continue, SE .14, built up over L14-L24) but weakly across contexts (held-out AUC .63-.79).

WHAT WAS CHANGED: model.layers.23.feed_forward.{w1,w2,w3}. Neuron 881 is freed by folding it into its near-duplicate (w2[:,331] += w2[:,881], cos .9996) and rebuilt as a gated unit. w1[881] = kap*(P w - theta*m/mp): w = unit shrinkage-LDA direction (lam 3) of corrupted-minus-clean inputs of the L23 MLP, fitted on half of the pass-1 questions (57 pairs) with clean steps and control positions as within-class scatter; P projects off the first prompt tokens; theta 1.70 = 99th percentile of out-of-fold negatives; m/mp = mean-input bias proxy. w3[881] = m/mp. w2[:,881] = unembedding of {Wait, Actually, Hmm, Hold, But, However} minus {So, Thus, Then, Now, Therefore, Hence, Next}, mapped through the norm gains, +3 logits at the median firing activation (column norm about .37, within the Q8 block limit). No training.

READ-PASS GATE ON THE TARGETED STEP (pass 6, questions NOT used to fit the gate, edited vs parent on the same texts): P(re-check opener) after a clean arithmetic step .217 -> .226 (+.009, SE .008); after a leading-digit error .327 -> .368 (+.041, SE .018); after a last-digit error (never used anywhere) .266 -> .310 (+.045, SE .018); after a mis-copied question number .129 -> .218 (+.088, SE .038), clean restatement .002 -> .002. Corrupted-minus-clean gap: +.032 (SE .016), +.036 (SE .016), +.088 (SE .038). Off-target over 184k teacher-forced tokens: mean |dlogp| .0020, 0.32% of tokens > .1.

PASS@4 GENERATE (6 passk questions x 4, cap 8,192, parent cached; measured with the WIDER version v1 = same unit, theta 1.41, not with the submitted one): overall 58.3 -> 75.0 (+16.7, SE 10.5, +5/-1); on both-finished answers (20) wrong->right 3, right->wrong 1 (+9.7, SE 10.6). Not evidence for the mechanism: 5 of the 6 questions were picked for parent failures, every edited trace diverges from the parent's within the first 0-17% (median 7%), not at an erroneous step, so wrong samples are re-drawn. On holdout48 ids 24-47 the same v1 gave overall -12.5 (SE 6.9, +0/-3: three finished-right answers became cuts), both-finish (19) +0.0 (+0/-0); re-check openers per 1,000 tokens +55%. v1 acts as a general re-check booster, so the threshold was raised (submitted version).

HOLDOUT48 AT 8,192 TOKENS (submitted version, passes 7-8): overall 62.5 -> 62.5, paired diff +0.00 (SE 2.98, +1/-1). Both-finish (40 questions): 75.0 -> 75.0, +0.00 (SE 3.58, +1/-1); 7 of 40 answers changed. Cuts 7 -> 7, mean tokens 3,472 -> 3,270, re-check openers per 1,000 tokens 4.09 -> 5.20; 4 of 48 traces identical, first difference at a median 8.8% of the trace.

EXPECTED ON SELECTION: about 0. The error the unit detects is rare in natural traces (about 5 consequential arithmetic slips in 2,021 checked statements; no mis-copies found), and finished-wrong answers are verbal judgement errors, not step slips. GSM8K was not measured; the unit fires after numeric statements, so the transfer guard is a real check for it.

## Selection (hidden questions) and decision

Parent accuracy on the 256 selection questions: **60.55%**. Rule: z >= 2.0; both-finish z >= 1.0; gsm8k_boxed guard z > -2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| ed_f/ed_f_g13_error_recheck_gate_L23 | 60.16 | -0.39 | 1.7 | -0.23 | 9 | 10 | +0.0000 | 8188 |
| claude/claude_g13_selfcheck_recheck_L23 | 59.77 | -0.78 | 1.8 | -0.43 | 10 | 12 | +0.0000 | 8191 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

_(no report yet)_

### ed_f

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

