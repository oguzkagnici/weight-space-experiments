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

## Pass 3 result: the model has a length counter in attention, but does not write it to the residual (code/010–011)
- **Sink heads.** Several full-attention heads put a large share of their attention on the first token. The share falls smoothly with absolute length N, as expected from sink dilution, e^D/(e^D+N).
  - Mean first-token weight: L18.H1 0.26, L18.H2 0.22, L18.H3 0.15; L18.H0, H5, H6 and H7 0.05–0.08; L2.H2 0.08.
  - The slope of log(1/w0 − 1) on log N is 0.6–0.85 for L18 H0/H3/H5/H6/H7 and L2.H2 (corr 0.7–0.76). A slope of 1 would be a pure sink.
- **Summed sink log-odds** over those 6 heads separates early (≤1,300 generated tokens) from late (≥2,300) positions with held-out d' 2.6; the oracle log N gives 4.6. So the attention pattern "knows" the length.
- **But it is not written out.**
  - L18 head outputs: d' 1.5. The best single head is L22.H4 at d' 1.9.
  - Residual h18, h20 and h22: d' 0.35–0.46. The content the heads average over swamps the sink term.
- This confirms pass 2: there is no readable late-length signal downstream. A counter would have to be built by re-reading a sink head's value through a new o_proj direction.
## Scope decision: no length-triggered wrap-up
- The brief (method, point 2) puts out of scope "edits that just shorten everything or end the thinking regardless of the problem".
- A gate on "late in the trace" is such an edit even if it gains (pass 1 suggests forced answers late are right 26–44%). I drop it.
## The carried gen-4 candidate wins selection clearly
- gate_wrapup_L20 on the 128 hidden selection questions (det1, paired): parent 48, edited 60, **+9.4 points, z 3.6, +12/−0**; cut 70 → 56.
- No trace is byte-identical. The repurposed neuron is slightly active everywhere, so every trace drifts, but all 12 flips are wins.
## Plan for passes 4–6: dose-response of the commit neuron
- DEV32 (code/013): all 9 cut-with-statement traces, 15 cut-without, 8 finished.
- Pass 4: carry exact on DEV32. Pass 5: a stronger write (target −15 instead of −9; w2 column × 15/9) on DEV32. Pass 6: a third dose or a lower threshold, depending on the result.
- Submit the best variant, so selection can test whether more push adds to the +9.4.
- Pass 4 (gen_005_claude_p04_6b1468): carry exact on DEV32. Pass 5 (gen_005_claude_p05_ffdb63): target −15 (w2 column 5414 × 15/9) on DEV32. Both are queued (passes snapshot the edit at submission).

## Passes 4–5 result: dose-response on DEV32 (code/015)
| edit | parent → edited correct | diff (SE) | +/− | cut | tokens |
|---|---|---|---|---|---|
| carry (target −9) | 8 → 9 | +3.1 (5.5) | +2/−1 | 24 → 20 | −85 |
| dose (target −15) | 8 → 9 | +3.1 (7.1) | +3/−2 | 24 → 19 | −151 |
- Comparing −15 with the carry directly: +3/−3. A stronger push adds nothing measurable at n = 32.
- By group:
  - cut with a statement: 0 → 1 for both; cut 9 → 5 for both;
  - cut without a statement: 0 → 1 (carry) and 0 → 2 (−15);
  - finished: 8 → 7 and 8 → 6 (one finished trace each became cut).
- **No trace stays identical.** The first differing character is at a median of about 300–550 characters, well before any commit point. The repurposed neuron, and the removal of its original role, perturb every trace a little. Under deterministic decoding that is enough to send each trace down a different path. So for this edit the "paired" comparison behaves like independent samples, and only large samples resolve it. That is consistent with the hidden selection set (n = 128, +12/−0) being far clearer than any 32-question test.
- The parent side reproduces the bank byte for byte (32/32), which confirms that the cache and engine are consistent.
## Pass 6: HOLDOUT32 (code/016) for the −15 dose
32 bank questions not in DEV32, drawn with a fixed seed. This dose is my candidate: selection will measure −9 (carry) against −15 side by side on 128 hidden questions, which gives a properly powered dose-response.
