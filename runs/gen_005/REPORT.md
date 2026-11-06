# Generation 5 report

- Created: 2026-11-06 07:32 UTC
- Parent: `base`
- Editors: claude, ed_a; budget 6 passes each; up to 1 candidates each
- Hidden selection subset: 128 questions (seed `selection-gen-5`)

## Suggested starting points

- **claude**: decision formation in truncated traces that never reach an answer: in finished traces, find the point where the model first commits to an option and what precedes it (which options were eliminated, what evidence was stated); in truncated traces of similar questions, find what is missing at the corresponding point. Look for components that mark "enough evidence to decide" or that keep an option open, and test whether strengthening the decision signal makes those traces commit to a correct answer (not just stop).
- **ed_a**: revisiting settled options: truncated traces deliberate option by option and often return to options they already ruled out or evaluated. Measure how often an option is re-evaluated after an explicit verdict on it, find the components active when the model reopens an option it already judged (heads attending back to the earlier verdict, MLP neurons), and test whether damping them saves tokens without reducing accuracy on finished traces.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 104 | read 104 texts | 1489.3 |
| claude | 2 | read | original | 14 | read 14 texts | 410.9 |
| claude | 3 | read | original | 44 | read 44 texts | 735.1 |
| claude | 4 | generate | both | None | read None texts | 1592.5 |
| claude | 5 | generate | both | None | read None texts | 1536.7 |
| claude | 6 | generate | both | None | read None texts | 1300.0 |
| ed_a | 1 | read | original | 64 | read 64 texts | 1684.2 |
| ed_a | 2 | read | edited | 64 | read 64 texts | 1570.4 |
| ed_a | 3 | generate | both | None | read None texts | 1771.4 |
| ed_a | 4 | generate | both | None | read None texts | 1666.9 |
| ed_a | 5 | generate | both | None | read None texts | 1282.6 |
| ed_a | 6 | generate | both | None | read None texts | 3010.2 |

## Candidates

### carry/gate_wrapup_L20

- Tensors: model.layers.20.feed_forward.w1.weight, model.layers.20.feed_forward.w2.weight, model.layers.20.feed_forward.w3.weight
- Weights changed (bf16): 6139

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

### claude/gate_wrapup_L20_d15

- Tensors: model.layers.20.feed_forward.w1.weight, model.layers.20.feed_forward.w2.weight, model.layers.20.feed_forward.w3.weight
- Weights changed (bf16): 6140

# gate_wrapup_L20_d15: the gen-4 "stop re-checking" neuron with a 15/9 stronger write

Edit (layer 20 MLP, neuron 5414, about 6,140 weights): identical to the carried candidate gate_wrapup_L20 (gate = the
"answer just stated" commit direction minus a threshold at its 99.8th all-token percentile, s > 10.28; up = bias proxy;
down = wrap-up minus reopen word direction mapped back through the final norm), except that the w2 column 5414 is
scaled by 15/9 (estimated logit-difference shift at commit points -15 instead of -9). w1 and w3 rows unchanged.

Purpose: a dose-response point. Selection measures the carry (-9) and this edit (-15) on the same hidden questions;
the carry scored +9.4 pp there (z 3.6, +12/-0).

Evidence (thinking mode, det1 engine, paired against the parent; trace-bank questions, parent side cached):
- DEV32 (9 cut-with-statement, 15 cut-without, 8 finished; used to compare doses): 8 -> 9 (+3.1, SE 7.1; +3/-2); cut 24 -> 19;
  the carry on the same DEV32: 8 -> 9 (+3.1, SE 5.5; +2/-1). -15 against the carry directly: +3/-3.
- HOLDOUT32 (32 other bank questions, fixed-seed draw, measured once): 17 -> 16 (-3.1, SE 3.1; +0/-1); cut 13 -> 14.
- Combined 64: +0.0 pp. I expect no gain over the carry; the stronger push did not add measurable benefit.
Note: no trace stays byte-identical (first difference at a median of 300-550 characters), so for this edit pairing behaves
like independent samples and small tests have little power.

### ed_a/ed_a_g5_churn_hence_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 6143

Targets HYPOTHESIS CHURN inside the thinking: at a paragraph break after an uncertainty statement ("Not sure.", "Hmm."), the parent opens yet another speculative hypothesis or method ("Alternatively, maybe ...", "Maybe ...", "Perhaps ...", "Another approach ...") instead of committing. On the 64 DEV trace-bank traces such churn openers follow 7.4% of paragraph breaks in cut traces vs 1.4% in finished ones (126 vs 12; paragraph-initial "Alternatively" 28 vs 1); the model's own P(churn opener) at breaks is .051 in cut vs .008 in finished traces. Paragraph-initial Hence/Therefore are pure commit moves (finished 22, cut 2; always followed by 'answer'/'final answer').
CHANGE (one repurposed neuron, L23 n74, the quietest L23 neuron; 2,048 weights in each of 3 tensors): model.layers.23.feed_forward.w1 row 74 = 16*(d - (12.20/mp)*mu_hat); w3 row 74 = 4.66*mu_hat/(16*mp); w2 column 74 = clip(2*u, +-0.98 * absmax of the other 31 weights of its Q8_0 block in each row). d = unit(unit(g1)+unit(g2)) at the L23 MLP input, g1 = mean at churn breaks - mean at other breaks, g2 = mean at churn breaks - mean over 60k ordinary tokens (recorded on the 64 DEV traces, pass 1); mu_hat = unit mean input (bias proxy, mean projection mp = 6.47); threshold 12.20 = 55th percentile of churn-break scores. u = unit(g_post23 * g_final * (mean E[Hence,Therefore] - mean E[Alternatively,Maybe,Perhaps,Another,Possibly])). The neuron is ~0 except above threshold (fires on 59% of churn breaks, 11% of other breaks in cut traces, 2.4% of breaks in finished traces, 0% of commit breaks and 0.000% of non-break tokens on DEV).
MECHANISM EVIDENCE (teacher-forced read on 64 DEV traces, v1 with the same gate): at firing churn breaks P(Hence/Therefore) .000 -> .39 and P(churn opener) .28 -> .055, graded by activation (act 2-4: P(H/T) .91); elsewhere mean |dlogp| 0.0004 (7 of 165k positions > 0.1). In generations, paragraph-initial churn openers fall 0.92 -> 0.56 per 1k tokens (DEV) and Hence/Therefore openers rise 0.10 -> 0.87.
MEASURED (generate both, paired, det1): DEV 32 (traces with a firing; used for choosing) +9.4 (SE 5.2, +3/-0); HOLDOUT 32 trace-bank questions 0.0 (SE 4.5, +1/-1); HOLDOUT 32 fresh random practice questions +9.4 (SE 5.2, +3/-0). Two holdouts combined (64): +4.7 points (SE ~3.4, +4/-1), cut share -1.6 points.
CAVEATS: replacing any neuron perturbs every logit slightly, so under det1 nearly all edited traces diverge early on ordinary words (no L23 neuron is dead or quantises to zero); paired differences therefore include resampling noise. A w2 column larger than its Q8_0 block peers made this much worse in an earlier version (block scale change); v3 keeps it under the block absmax. Some firings produce "Hence we need to ..." (connective, not a commit). The gate acts mainly in knowledge questions (law, health, psychology, philosophy, economics); computation-heavy cut traces rarely trigger it.

## Selection (hidden questions) and decision

Parent accuracy on the 128 selection questions: **37.5%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| carry/gate_wrapup_L20 | 46.88 | +9.38 | 2.6 | +3.61 | 12 | 0 | +0.0000 | 6139 |
| claude/gate_wrapup_L20_d15 | 43.75 | +6.25 | 3.1 | +2.02 | 12 | 4 | +0.0000 | 6140 |
| ed_a/ed_a_g5_churn_hence_L23 | 40.62 | +3.12 | 1.9 | +1.64 | 5 | 1 | +0.0000 | 6143 |

**Decision:** winner `carry/gate_wrapup_L20` becomes the new champion.

## Editors' reports

### claude

# Report of editor `claude`, generation 5

## What the model gets wrong
- **Running out of budget is the main failure.** 52 of the 96 bank traces hit the 3,072-token cap.
- **Most cut traces never state an answer** (43 of 52). They give fewer verdicts on options (0.39 per 1k tokens, against 0.59 in finished traces) and spend their budget on:
  - long numeric work with self-doubt and repeated unit conversions;
  - re-reading what the question asks;
  - judging the same options again.
- **These traces have not converged on an answer** (pass 1, a forced-answer probe at 30% and 65% of each cut trace).
  - The forced argmax is correct only 19–26% of the time (chance is 10%), with a mean confidence of 0.8.
  - Only confidence ≥ 0.95 is informative (41% correct).
  - The forced answer changes between the two probe points in half of the traces.
  - So making them "commit when confident" cannot rescue them.

## How the model tracks length (passes 2–3)
- The residual stream has no usable "how long have I been thinking" signal.
  - A late-versus-mid direction reads wrapping-up content and saturates by about 1,500 tokens.
- Sink-dilution heads do track absolute length in their attention pattern (summed log-odds d' 2.6 between ≤1,300 and ≥2,300 tokens).
  - The heads are L18 H0, H3, H5, H6 and H7, and L2.H2.
  - Their outputs mix that term with the content average, so the residual carries almost none of it (d' ≈ 0.4).
- I did not build a length-triggered wrap-up from this. The brief puts out of scope edits that end the thinking regardless of the problem.

## What I changed and why
- **The carried gen-4 neuron, `gate_wrapup_L20`, won selection clearly.** On the 128 hidden questions under the deterministic engine it scored +9.4 pp (z 3.6, +12/−0).
- **I tested a stronger write as a dose-response:** neuron 5414's w2 column scaled by 15/9, target −15 instead of −9.
- **I submitted that dose, `gate_wrapup_L20_d15`,** so that selection measures −9 and −15 side by side.

## Evidence
| set | edit | parent → edited | diff (SE) | +/− |
|---|---|---|---|---|
| DEV32 | carry (−9) | 8 → 9 | +3.1 (5.5) | +2/−1 |
| DEV32 | dose (−15) | 8 → 9 | +3.1 (7.1) | +3/−2 |
| HOLDOUT32 | dose (−15) | 17 → 16 | −3.1 (3.1) | +0/−1 |

- The stronger push adds nothing measurable. I expect the carry to beat my candidate in selection.
- **No edited trace stays byte-identical.** The first difference comes at a median of 300–550 characters, long before any commit point.
  - The repurposed neuron (its old role removed, its new gate slightly active everywhere) nudges every trace, and deterministic decoding then follows a new path.
  - So "paired" tests of such edits behave like independent samples. Only large samples, like the 128 selection questions, resolve them.

## What I would try next
1. **A cleaner commit gate.** Off-target drift makes every comparison noisy and may cost finished traces. Two ways to reduce it:
   - put the gate on a neuron whose original output is near zero on ordinary tokens;
   - lower the up-row (bias proxy) magnitude, so that the sub-threshold silu leakage times the up value is negligible.
   - Check first with a read pass along finished traces that the per-token log-prob change is about 0 before the first commit point.
2. **The never-concluding traces (~80% of cut traces) need a reasoning-level target, not a stopping rule.** Two candidates:
   - the option-revisiting loop (`ed_a`'s topic);
   - the repeated unit-conversion and self-doubt episodes in numeric problems.
   - Pass-1 data (`work/probe_rows.json`) gives the forced belief per trace for choosing them.
3. **If a budget-aware model is ever in scope,** a counter can be built by re-reading a sink head's first-token value through a dedicated o_proj direction. The sink heads are listed above.

### ed_a

_(no report yet)_

