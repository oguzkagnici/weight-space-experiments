# ed_a, generation 1: removing the answer-letter position prior

## What the model gets wrong
- **Letter-position prior.** On 512 DEV questions the parent (33.6%) over-picks early letters and F, and rarely picks late letters. Predicted counts were A85 B99 C76 F67 H23 I20 J14; the true counts were A69 B48 C51 F55 H52 I41 J49. The mean centered letter logit on 10-option questions is about [A 1.2, B 1.3–1.7, C 0.6–1.2, D 0.2, E 0.1, F 1.2, G −0.2, H −0.9, I −2.1 to −2.6, J −1.6]. This pattern held on every question set I scored. On 4-option questions the prior is [.64 .34 .33 −1.31], so D is under-picked.
- **Size of the problem.** Subtracting a perfect constant per-letter bias offline (2-fold CV) would raise accuracy from 33.6% to about 37.5%. That is the ceiling for any pure bias fix.
- **Near misses are common.** Among wrong answers, the correct letter ranks second in 25% of cases and third in 18%.
- **Where the answer letter gets written.** It only becomes readable in the residual stream at layers 22–23. Head **L22h12** is the answer-copy head: it adds +2.96 logits to the correct letter. It has its own bias (G +1.6, B −1.4).
- **The L22 MLP suppresses the copied letter.** It has a bank of one "retrieved-letter suppressor" neuron per letter: A389, B6620 and B5859, C1812, D464, E7357, F6492, G3344, H397, I2786, J3310. Each fires when its letter is L22h12's top candidate and pushes that letter down. The push is uneven: A −1.5, B about −1.5 (both neurons together), C −1.9, D −3.7, E −4.7, F −3.0, G −3.1, H −2.4, I −3.2, J −2.9 logits. L23 has a second, weaker and more even bank (L23 n3631, n7209, n1192, …).

## What I changed and why
- **Edit v2 (primary candidate).** For 27 MLP neurons in layers 12–23 and head L22h6, I projected the 9-dimensional centered letter-readout subspace (span of wn⊙E[A..J] minus their mean) out of each one's output column. This is done in post-norm-scaled space, at full strength.
- **How the components were chosen.** They are near-constant at the answer position (coefficient of variation below 0.6), carry almost no answer content, and their mean direct-logit contribution lies along the bias. The largest are L20 n1000, L22 n190, L17 n2401 and n616, and L23 n1047, n4597 and n1903.
- **Why this is a targeted fix.** It removes what writes the fixed position prior and leaves the content path (L22h12 and the suppressor banks) untouched.
- **Edit v1 (second candidate)** is the 12-neuron subset of v2.

## Evidence
All gains below are paired (edited minus parent).

| Edit | Question set | Accuracy gain (pp) | Log-prob gain on correct letter |
|---|---|---|---|
| v1 | Screening set S = DEV[512:768] (256 questions) | +2.34 ± 1.35 | +0.190 ± 0.041 |
| v2 | S | +3.12 ± 1.55 | +0.247 ± 0.047 |
| **v2** | **HOLDOUT, 768 questions never used for choosing** | **+1.69 ± 0.96** | **+0.183 ± 0.026** |
| v1 | HOLDOUT[0:256] | +1.17 ± 1.61 | +0.138 ± 0.039 |
| v2 | HOLDOUT[0:256] (same questions) | +1.95 ± 1.61 | +0.188 ± 0.044 |

- **The prior flattens as predicted.** On 768 questions (S plus HOLDOUT) the residual bias goes from [1.25 1.34 .99 .21 .14 1.28 −.2 −.88 −2.33 −1.8] to [.3 −.1 −.13 −.13 .01 .78 −.26 −.06 −.04 −.37]. That recovers about 75% of the gain available from the offline debias ceiling.
- **No self-repair.** The per-letter shift that v1 actually produced matched the direct-logit prediction from pass-1 activations.
- **Dose-response is smooth.** In the direct estimate, strength 0.25 / 0.5 / 0.75 / 1.0 / 1.25 gives 35.7 / 36.5 / 37.1 / 36.9 / 36.5% accuracy.
- **Transfer to fewer options.** On the 151 four-option practice questions (GPQA has 4), accuracy goes from 45.7% to 47.7% and the D under-pick roughly halves (−1.31 to −0.74). On all 195 questions with 6 or fewer options the gain is +1.54 ± 0.88, but the log-probability of the correct letter falls slightly (−0.05 ± 0.03).

## Negative results
- **Halving the L22 suppressor bank** (all 11 w2 columns ×0.5) gave +0.78 ± 0.78 pp and a log-probability change of −0.05. The direct-logit estimate had predicted about +3 pp. Downstream layers (most likely the L23 bank) undid most of it, and what remained just raised whichever letter was already on top, which only sharpens the output.
- **Equalizing that bank on top of v2** (each letter suppressed like A) did slightly worse than v2 on S: +2.73 versus +3.12.
- **Lesson:** direct-logit estimates are reliable for constant-bias edits, but not for edits to the conditional suppressors in L22.
- **More bias-removal components** (more neurons, or all low-variability neurons) add nothing in the direct estimate. The bias route is close to its limit with v2.

## What I would try next
- **Content, not bias.** The remaining headroom is in content. Head L22h12 decides most answers, so I would study what it attends to (a read pass with `attn_layers=[22]`) and why it under-retrieves A and B and over-retrieves G. This may be a position or recency term in its keys or queries.
- **Leftover F bias.** v2 still leaves F at +0.78. Its sources are spread over L21 and L23 attention and the L23 MLP.
- **Near misses.** Because the correct letter is often second, test whether lowering the suppressor banks together in L22 and L23 helps once the prior is removed. Measure it on held-out questions, since the L22-only version was compensated downstream.
