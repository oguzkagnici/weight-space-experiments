# Report of editor `claude`, generation 3

**Starting point: the writers of the belief in layers 8–14** (refuted).
- I recorded the residual stream at every token for 192 DEV questions and read it at the last content token of each option line.
- A fixed correct-vs-incorrect direction there picks the gold option only 16–22% of the time (chance 11.5%), for both a mean-difference and a logistic probe, cross-validated by question. The model itself gets 34–38%.
- No single block writes much of it.
- The model's choice is therefore relational, a question-dependent query against the option keys of L22.H12, not a fixed "this option is right" feature. Amplifying fixed belief writers has nothing to amplify.

**Pivot: re-test the best hidden candidate so far, with a dose-response.**
- Generation 2's ablation of L12.H13 (`ed_b`) had hidden +1.25 (z 1.04), gold rank −0.11 and dlogp +0.50.
- I rebuilt it from its description and tested ×0 and ×0.5.
- Chosen on DEV 512:

  | Edit | Accuracy | Gold rank | dlogp |
  |---|---|---|---|
  | ×0 | −1.37 (SE 1.45) | +0.08 (worse) | +0.40 |
  | ×0.5 | +0.39 (SE 0.87) | 0.00 | +0.19 |

- HOLDOUT, measured once:

  | Edit | Questions | Accuracy | Gold rank | dlogp |
  |---|---|---|---|---|
  | ×0.5 | 768 | +0.39 (SE 0.65) | −0.048 (SE 0.026) | **+0.262 (SE 0.020)** |
  | ×0 | 512 | −0.20 (SE 1.40) | −0.10 | +0.535 |

- The generation-2 accuracy and rank gains of the full ablation do not replicate on my data, which looks like a best-of-five effect. The calibration gain replicates everywhere and scales with dose.
- Candidates:
  - `l12h13_half` (primary; chosen on DEV);
  - `l12h13_ablate_retest`, submitted so a fresh hidden subset re-tests generation 2's result without bias.

**What three generations show.** In direct-answer mode, every editor found real mechanisms:
- the letter-prior writers;
- the uneven leader suppressors;
- the option-primacy query component;
- the router weights;
- the confidence writer L12.H13.

Edits to them reliably change confidence and the distribution of predicted letters, but accuracy gains have not survived fresh questions. The errors are content errors: L22.H12 attends to the wrong option because the question-to-option match is wrong.

**What I would try next.**
1. The one untapped signal is the option-text likelihood (cloze; `ed_a` this generation).
2. Otherwise, accuracy may be near what weight edits can reach in this mode. A calibration objective, or the thinking-mode protocol on a GPU, would be the natural next steps; the latter is where the paper's obstacle-removal effect lived.
