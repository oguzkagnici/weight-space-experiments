# ed_b, generation 3: report

## Summary
I followed the suggested starting point, the content side of L22.H12, and it came out **negative**. Given the keys it reads, H12's query is already close to optimal. Its errors come from the option-end "beliefs", and those build up steadily through layers 14–22 without being lost later.

I found one new, content-free influence: H12 and most other L22 heads attend to the option **letter tokens** "(X". Removing it is accuracy-neutral on HOLDOUT, with a small calibration gain that replicated.

I also found that on 512 DEV questions, a perfect constant per-letter debias gains **+0.1 points**. The letter bias is real but costs almost no accuracy. That is consistent with every letter-bias edit from generations 1–2 scoring about 0 on the hidden questions.

## What the model gets wrong
**H12 is the answer, and its readout is saturated.** On DEV[:240], pass 1, code/003–015:
- H12's attention argmax picks gold 39.2%, against 39.6% for the final answer; they agree 80% of the time. There is no gap between H12's choice and the output left to recover.
- The query is about 93% a constant vector (cosine with the mean). A constant query alone scores 34%, so the question-specific part is worth about 5 points. Amplifying it ×1.5–3 gains nothing.
- Adding the mean gold-minus-wrong key direction to the query hurts in cross-validation (.396 → .30).
- Removing query directions with a negative mean contribution to the gold margin gains nothing (13 wrong→right / 13 right→wrong).
- Per-slot debiasing of H12's scores gains nothing (.374 → .364). Projecting out the slot-mean key subspace over-corrects: A is never picked.
- Other L22 heads pick as well as H12 (H14 .404, H15 .396) but agree with it 81–88% of the time. Combining their votes gains at most 5 of 240.
- Sharper H12 attention would help a little: temperature ×2 gives +8/−2 in the lens. But q_norm and k_norm are shared across heads, the RMS-norm epsilon plays no role, and only about 17% of the query lies in the key null space. A per-head temperature therefore cannot be edited cleanly; the most available is ×1.10.
- The value H12 copies decodes to the right letter 97–99.6% of the time at every option length.

**Knows-then-loses does not happen at option positions.** Pass 3 (code/021–022) used a key lens: L22's own norm, k_proj and k_norm applied to earlier residual layers, scored with H12's real query.
- H12's gold pick rises steadily with the layer the keys are read from: h10 .16, h12 .20, h14 .28, h16 .38, h18 .38, h20 .38, h22 .43.
- Scaling any single block's contribution moves it by at most ±3 of 120.
- The belief is distributed, and it is simply wrong on the questions the model misses.

**Letter stage** (pass 2, DEV2 = 512 questions, exact direct logit attribution, code/016–019):
- Mean centred letter logits on 10-option questions, by stage:

  | Stage | A | B | C | D | E | F | G | H | I | J |
  |---|---|---|---|---|---|---|---|---|---|---|
  | h22 | 1.14 | 1.16 | .18 | .09 | .45 | −.04 | −.70 | −.79 | −.81 | −.69 |
  | H12 (own tilt) | −.59 | −1.41 | .68 | −.50 | −.21 | .26 | 1.18 | .18 | .09 | .33 |
  | mlp22 (gold-help −1.14) | .8 | .78 | .23 | .4 | −.17 | −.17 | −.41 | −.34 | −.22 | −.89 |
  | L23 | .13 | .44 | −.43 | .02 | .18 | .98 | −.26 | .04 | −.26 | −.83 |

  H12's gold-help is +2.86. Generation 2 could not explain the F preference; layer 23 writes it (F +.98).
- Predicted letters A..J: 82 93 98 54 41 44 36 26 21 17; true: 64 63 54 70 39 47 51 45 42 37.
- Yet a cross-fitted constant debias gains +0.1 points (SD .7 over 20 splits).
- There are no off-by-one (neighbouring option) errors and no format anomaly in any category.

## What I changed and why
**Mechanism.**
- L22 has no positional rotation; the dot product of the recorded `q_norm` and `k_norm` outputs reproduces its attention.
- H12 puts .116 of its attention on the "(X" letter tokens. These come *before* the option text, so by causality their keys carry a slot code and no option content.
- In the lens, dropping letter tokens from H12's option vote gives:
  - +8/−0 with q·k recomputed (B over-pick 36 → 26);
  - +5/−2 with the recorded attention.
- Most other L22 heads also attend to letter tokens (.07–.50 of their attention), and dropping those tokens improves their picks too (H3 +25/−5, H11 +14/−4).

**Edit** (code/027 and 028). Only `model.layers.22.self_attn.q_proj.weight` changes:
- For each edited head: `W_h <- W_h - a w wᵀ W_h` (rank 1 per head).
- `w = normalize(q_norm.weight * u_g)`.
- `u_g` = mean over 240 DEV questions of (key at the letter tokens − key at the option end and newline tokens), for kv group g.
- q_norm renormalizes the query afterwards.

**Candidates.**
- `letterproj_L22_h12`: head 12 only, a = 1.5. In the lens this cuts letter-token attention from .116 to .026.
- `letterproj_L22_all`: all 16 heads at a = 1.0, head 12 at 1.5.

## Evidence
All paired, edited − parent.

| Edit | Questions | Accuracy (pp) | Wrong→right / right→wrong | Log-prob of correct | Gold rank |
|---|---|---|---|---|---|
| H12, a 1.5 | DEV3 (256) | +1.17 (SE .87) | 4/1 | +.022 (.023) | −.055 (.037) |
| All heads, a 1.0 | DEV3 | +0.39 (1.17) | 5/4 | +.040 (.023) | −.078 (.039) |
| All heads, a 1.5 | DEV3 | 0.00 (.96) | 3/3 | +.065 (.023) | −.055 (.037) |
| **H12, a 1.5** | **HOLDOUT[0:512]** | **0.00 (~.73)** | **7/7** | +.020 (~.017) | −.025 (~.027) |
| **All heads, a 1.0** | **HOLDOUT[0:512]** | **0.00 (~.74)** | **7/7** | **+.047 (~.017)** | −.031 (~.027) |

Order in which I looked at results:
- I chose both candidates on DEV3 and wrote that choice in notes.md before any HOLDOUT pass.
- I then measured H12-only on HOLDOUT (passes 7–8) and the all-heads edit afterwards (passes 9–10). Each was measured once.
- I chose nothing after seeing HOLDOUT.

The accuracy change is null. The log-prob gain of the all-heads edit replicates (DEV +.040, HOLDOUT +.047, z ≈ 2.8) and grows with strength. It is a calibration gain, not a ranking gain.

## What I would try next
- Selection, letter and routing edits all seem capped at about ±1 point. Stop working on L22 and the letter stage.
- The errors are in the option-end beliefs (layers 14–22, distributed). The only unused knowledge found so far is the option-text likelihood: generation 2's cloze test picked gold on 25% of the questions the model gets wrong.
- An edit would need to route a "surprise/likelihood" signal into the option-end keys. That is a new circuit, and I see no cheap handle on it.
- If a cheap target is still wanted: L22 attention sharpness has a monotone lens effect (T 1.5 / 2 → +4/−1 and +8/−2 on H12's pick). It can only be changed layer-wide, through the shared `q_norm`/`k_norm` gains, so its effect on the other heads would have to be measured.
- For GPQA transfer, the 4-option D under-pick (generation 1) is unaddressed.
