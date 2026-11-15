# Report: editor claude, generation 12

## What the model gets wrong
I tested "knows but does not say": does the network prefer the right option somewhere inside, with a later component overriding it?
- **It does not.**
  - **Setup:** forced-answer probes ("So far, the most likely answer is (") at 35%, 50%, 70% and the end of the thinking.
    - 20 wrong samples of the 15 mixed questions (pass 1).
    - 28 wrong traces of 13 other questions (pass 2).
    - Right-trace controls.
  - **Letter lens, residual h14–h24:** when the sample ends wrong, gold is never the top letter at any layer (≤ 2 of ~55 probes at any single layer), and P10(gold) ≤ .07. In right traces the lens picks gold 57–80% from h15 on.
  - **Attribution:** the only components pushing toward gold in wrong probes are the generic negative-feedback ones (L22 MLP leader suppression, L22 H11/H13/H14).
- **The readout copies the trace's stated belief.**
  - In thinking mode the current belief is readable from h15 (after L14). In direct mode it appears only at L22.
  - At the probes, L22 H12 attends to the pick's option line 3–5× more than to the gold line, and to letters the trace has already stated.
  - So wrong conclusions are wrong all the way through. This agrees with generations 2 and 10 ("missing knowledge").
- **Smaller checks:**
  - Questions where the pre-thinking prior is right but thinking ends wrong (6) are mostly lost to cuts (3 of 4 or 4 of 4 samples cut at 8,192), not to content.
  - Negation questions are rare (6%).
  - Stated hypotheses come late; when they equal the prior, the result is no worse.
- **The one measurable lever is the readout's letter bias.** The pre-thinking prior (forced answer right after `<think>`) is 29.7% accurate on 320 practice questions and badly letter-biased: it predicts A 87× against 46 golds, and J 2× against 21. Gen 11 found that wrong samples' early latent belief anchors on this prior.

## What I changed and why
- I added a constant per-letter correction to the forced readouts (`claude_g12_readout_letter_debias_L23`). It is carried by six L23 neurons that are reliably active at those readouts; n7528 and n6877 are silent at final-answer copy positions.
- Correction: c_L = log(mean P_L) − log(gold share), at full strength. It is written as the minimal-norm w2-column change that shifts the centered letter logits by −c at the readout.
- **Gate (read pass, held-out, paired against the parent):**
  - 64 pre-thinking prior probes (not used to fit c): P(gold) .282 → .313 (+.031, SE .012); accuracy 26.6 → 32.8% (+5/−1).
  - Mid-trace probes on mixed questions: +.068 (SE .006), +25/−4. This is overstated, because these questions' golds skew to late letters.
  - Off-target: top-1 changed at 1.85% of teacher-forced positions; actual-token log-prob −0.001 on average.

## Evidence on conclusions

| Set | Overall paired diff (SE), +/− | Both finished |
|---|---|---|
| DEV (6 mixed questions × 4 samples, 8,192) | +20.8 (13.4), +8/−3 | +20.0 (14.5), +4/−1 |
| **HOLDOUT48, 8,192** | **+0.00 (5.16), +3/−3** | **n 39, +0.00 (5.20), +2/−2** |
| 3,072 holdout bank (96) | +0.00 (3.63), +6/−6 | n 45, +2.22 (2.22), +1/−0 |

- Final letters stay consistent with the reasoning text (I checked the changed answers).
- Expected selection result: about 0. The readout gain is real, but conclusions are produced in the reasoning text, and this edit does not change that text's quality.

## What I would try next
1. **Stop targeting the answer path.** Four generations of evidence (gens 2, 10, 11 and 12) say the readout is faithful and wrong conclusions carry no latent gold. Any gain has to change what the reasoning text says.
2. **Cut answers on prior-right questions.** These are the clearest "knew it, lost it" cases, and they are lost by running out of budget, not by content. A content-aware loop breaker (detect repeating re-derivations and conclude with the current leader) would act exactly there. Its selection gain would mostly appear in the overall diff, not on questions both models finish, so it needs the selection rule's view on cuts.
3. **Pass@k vs single-sample variance** (majority vote +5.7 points) is the remaining content headroom. Read passes cannot reach it, because conclusions form gradually in the text. It needs generate-pass designs with more samples per question than 8 passes allow.
