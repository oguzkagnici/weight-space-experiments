# Notes of editor claude, generation 5

## Diagnosis on the det1 trace bank (code/001, 002)
- Groups: finished 44 (37 correct, 1,544 tokens); cut with an in-think answer statement 9; **cut without a statement 43** (law 9, physics 7, health 6, economics 5, engineering 4).
- Cut traces without a statement mention 4.3 distinct options and make **fewer verdicts** (verdict phrases 0.39 per 1k tokens) than finished traces (0.59) or cut traces with a statement (0.76).
- The excerpts show varied failure:
  - long numeric work with self-doubt and repeated unit conversions (engineering, physics, business);
  - re-reading what the question asks ("maybe they ask for ...");
  - judging the same options again (other, law).
## Pass 1 plan: forced-answer probe (read, original)
- 52 truncated traces (43 without a statement, 9 with one), cut at the paragraph break before 30% and 65% of the trace, followed by "\n\nSo the answer is (". That gives 104 texts and 168k tokens; top-50 tokens are recorded at the last position.
- Question: does the model already hold a confident, correct belief mid-trace? If so, an edit that commits when confident could rescue these traces. If its forced belief is near chance, a stop-earlier edit cannot help, and the target has to be how it reasons.

## Pass 1 result: forced-answer probe (code/004)
| group | point | n | letter mass | argmax acc | mean conf | conf >= 0.8: n / acc |
|---|---|---|---|---|---|---|
| cut, no statement | 30% | 43 | 0.54 | 0.19 | 0.80 | 23 / 0.30 |
| cut, no statement | 65% | 43 | 0.55 | 0.26 | 0.78 | 25 / 0.32 |
| cut, statement | 30% | 9 | 0.78 | 0.22 | 0.79 | 5 / 0.40 |
| cut, statement | 65% | 9 | 0.94 | 0.44 | 0.90 | 7 / 0.43 |
- Calibration across all 104 points: confidence 0.4–0.95 gives 10–19% accuracy (near chance); confidence >= 0.95 gives 41% (n 39).
- The forced argmax is the same at 30% and 65% for only 25 of 52 traces, so the belief drifts.
- **Reading.** The model mostly does not hold a correct belief mid-trace (19–26%, against 10% chance). But a cut trace scores 0, so any commit has positive expected value on it (about +0.25 per cut trace forced at about 2/3 of the budget).
- The cost is in finished traces that an edit stops early. Long finished traces are already poor:
  - finished beyond 2,200 tokens: 12, of which 4 are correct (33%);
  - beyond 1,800 tokens: 21 (11 correct).
- **Idea: a "late wrap-up" edit.** It should commit only late in the trace (after about 2,200 tokens), not on confidence.
  - Rough expected gain on the bank: 52 cut × about 0.25 = about +13 correct, against about 0 net on the 12 late finishers. That is about +10 points if the late trigger is clean.
  - It needs a length signal inside the network.
## Pass 2 plan: is there a linearly readable length signal? (code/005)
- Read the residual (layers 14 and 20) at all positions on 7 cut traces and the 7 longest finished traces (43.6k tokens).
- Compute the difference of means between late (>2,200) and mid (500–1,500) positions, cross-validated over traces; check the score at "\n\n" tokens.

## Pass 2 result: no clean length signal in the residual (code/006–008)
- The late-minus-mid direction, pooled over all 14 traces, rises monotonically with position (L20: −4.7 at 0–500 tokens up to +4.6 at 2,500+; held-out AUC 0.75 at paragraph breaks). But a threshold set on early breaks fires only in **finished** traces near their end, never in cut traces. It reads "wrapping up" content, not length.
- Refit on cut traces only (leave-one-cut-trace-out), the score saturates by about 1,500 tokens (L20 by bin: −6.8, −1.2, −0.4, +3.1, +1.3, +3.9). Finished traces score just as high from 1,000 tokens on, so it cannot separate 1,500 from 2,500 tokens.
- **Conclusion.** Under a difference-of-means read, the residual carries early-versus-later (the first ~1,000 tokens) but no usable late-length signal. A gate on "late in the trace" has to come from a component that integrates over the context.
## Pass 3 plan: look for a counter head (code/009)
- In an attention head with a strong sink on the first token, the sink weight is e^Δ/(e^Δ+N). That decays smoothly with absolute position N and averages out content noise.
- Read prefixes of 10 traces cut at paragraph breaks near 600, 1,200, 1,800, 2,400 and 2,900 generated tokens.
- Record attn_last (all full-attention layers) and the per-head o_proj inputs at the last token (layers 14, 18, 22). Then look for heads whose first-token weight or output tracks N with low spread across traces.
