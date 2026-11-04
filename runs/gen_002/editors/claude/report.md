# Report of editor `claude`, generation 2

**Starting point: near misses.** I reused my generation-1 recordings (same parent, same 512 DEV questions).
- The correct letter is ranked second in 76 of 512 questions. These are not close calls: the median probability gap is 0.44, and only 17% are within 0.1.
- In near misses, L22.H12 gives the wrong winner +2.8 logits, against +6.6 to the correct winner in near wins. The L22 MLP suppresses whichever letter wins, in proportion.
- The winner is the earlier letter in 57% of near misses, against 38% of near wins. That is the generation-1 letter bias again.
- Conclusion: near misses are H12 picking the wrong option, not a late component flipping a close call.

**Hypothesis H5: an ensemble of routers.**
- Besides H12 (own-choice accuracy .371), heads L14.H15, L14.H12 and L20.H7 write weak option votes: .26–.28 own-choice accuracy against about .11 chance.
- They agree with H12 only 31–36% of the time and are right on 15–16% of H12's mistakes.
- Their direct effect is small (about 0.1 logit). I amplified their outputs (o_proj columns) so the ensemble gets more say.

**Results** (paired; candidates chosen on DEV before any HOLDOUT measurement):

| Edit | DEV | HOLDOUT (measured once) | dlogp(correct) |
|---|---|---|---|
| Routers ×1.5 | +1.17 (SE 0.87) and +1.17 (SE 1.30) on two DEV sets | 768 q: about +0.8 (SE ~0.55) | −0.14 |
| Routers ×2 | +1.95 (SE 1.51) | 512 q: +0.20 (SE 1.09) | −0.30 |
| Routers ×3 | +1.95 (SE 1.70) | (not measured) | −0.55 |
| Routers ×1.5 + gen-1 `equalize_boost` | +1.56 (SE 1.66), fresh DEV | (not measured) | −0.21 |

- H5 is at best a small effect. The weak routers' votes don't carry enough independent correct signal to rescue H12's mistakes.
- Like generation 1's suppressor edits, the gains come with a calibration cost: amplifying the routers increases confidence overall.
- Candidates: `routers_x1p5` (primary) and `routers_x2`.

**Conclusion across both generations.**
- In direct-answer mode, LFM3-1.7B's errors are mostly H12 attending to the wrong option, a content and knowledge failure.
- The removable obstacles found so far (the letter prior and router weighting) are each worth about 1 point or less on accuracy.
- This differs from the paper's Qwen case, where a single mechanical failure (looping) blocked most answers.

**What I would try next.**
1. Measure head 12's attention to the correct option's span versus its chosen option's span, using `attn_last`. Check whether its errors correlate with a non-content factor such as recency or option length; this is ed_a's direction in this generation.
2. Since accuracy gains are hard to get, consider making calibration (correct-letter log-probability) a secondary objective. Several edits significantly improve it without hurting accuracy (generation-1 `ed_a` v2: +0.18).
