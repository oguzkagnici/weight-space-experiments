# Gen 4: claude editor report (thinking mode)

**Candidate:** `gate_wrapup_L20`. One repurposed MLP neuron (layer 20, neuron 5414: w1 row, w3 row, w2 column; 6,139 weights).

## Hypothesis
In thinking mode, 53% of base traces hit the 3,072-token cap. In those, 17 of 53 had already stated an answer inside `<think>` (9 of them correct) and then kept re-checking.
Across the 43 finished traces, re-checking after the first statement changes the answer only 2 times.
So a neuron that, right after a commitment, shifts the next-token choice from "reopen" words (Wait, But, Let me...) to "wrap-up" words should turn cut traces into correct finished ones at little cost.

## What was found about the model
- There is no single component that pushes reopening. Direct-logit attribution over attention heads and MLPs after a commitment is diffuse.
- The commit state is linearly readable from the layer-20 residual (cross-validated AUC about 0.87). Its score is +13.4 at commit points, -2.0 at controls and about 0 at random tokens.
- A gated neuron built from those means fires on 71% of commit points and 10% of controls (about 0.2% of all tokens).
  At commit points that reopened, it moved P(reopen) 0.81 -> 0.68 and P(close) 0.11 -> 0.24.
  The real effect is about a third of the linear estimate (later layers partly cancel it).

## Passes (6 of 6 used)
| pass | kind | set | result |
|---|---|---|---|
| 1 | read | 96-trace bank, decision points | DLA: diffuse; commit signal readable |
| 2 | read | commit and control points | residual reconstruction validated (corr 0.99999); commit score separates |
| 3 | read, edited (target -3) | commit and control points | P(reopen) 0.81 -> 0.68; controls unchanged |
| 4 | generate both (target -9) | DEV 32 (tuning) | 37.5 -> 46.9 (+9.4, SE 6.9; +4/-1); cut 0.53 -> 0.31 |
| 5 | generate both | HOLDOUT A 32 | 31.2 -> 34.4 (+3.1, SE 3.1; +1/-0) |
| 6 | generate both | HOLDOUT B 32 | 9.4 -> 18.8 (+9.4, SE 5.2; +3/-0); cut 0.81 -> 0.69 |

The two holdouts combined (64 questions) give +6.3 pp (SE about 3.0; +4/-0).

## Expected selection effect and risks
- The gain comes only from traces that commit and then over-check. Traces that never state an answer (most cut traces in HOLDOUT A) are untouched. I expect +2 to +6 points on the hidden selection set.
- Finished traces get shorter, typically by 300–1,400 tokens, with the same answer in every holdout case. The one DEV loss (mmlupro_9424) was a finished trace that diverged.
- Direct-answer mode (phase 1) is not measured. The neuron is gated on a state that only occurs mid-thinking, so it should be inert there; this needs checking on GPQA and the phase-1 held-out set.
