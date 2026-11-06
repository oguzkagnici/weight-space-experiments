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
