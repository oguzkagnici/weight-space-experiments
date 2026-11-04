# Generation 1 report

- Created: 2026-11-04 19:30 UTC
- Parent: `base`
- Editors: claude, ed_a, ed_b; budget 10 passes each; up to 2 candidates each
- Hidden selection subset: 600 questions (seed `selection-gen-1`)

## Suggested starting points

- **ed_a**: Answer-letter preference. The model may prefer some option letters (positions) regardless of content; measure this on practice questions (predicted vs true letter counts, and the letter probabilities on questions it gets wrong). If it is there, find what writes the position preference into the answer position (heads, MLP neurons, a constant bias direction, or a dependence on where each option sits in the prompt) and remove it in a content-preserving way. A uniform prior shift tuned to 10 options may not transfer to 4-option GPQA, so prefer fixing the mechanism.
- **ed_b**: Near misses and late-layer loss. When the model is wrong, check how often the correct letter is its second choice. Use the logit lens (residual stream after each layer, through the final norm and the tied output layer, restricted to the option letters) to find whether the correct answer leads at some middle layer and is lost later, and which late components (MLP neurons, attention heads) overwrite it.
- **claude**: How the answer is routed from the option text to the answer position: which attention heads at the last token attend to the option lines, whether they select by content or by position, and whether some heads systematically move the wrong option.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | score +acts | original | 512 | acc 38.09 | 462.6 |
| claude | 2 | score +acts | original | 512 | acc 38.09 | 380.6 |
| claude | 3 | score | both | 256 | paired +1.56 (SE 0.95), +5/-1, dlogp -0.1413 | 406.3 |
| claude | 4 | score | both | 256 | paired +1.56 (SE 1.23), +7/-3, dlogp -0.2981 | 187.2 |
| claude | 5 | score | both | 256 | paired +1.17 (SE 1.41), +8/-5, dlogp -0.4339 | 166.7 |
| claude | 6 | score | both | 256 | paired +2.34 (SE 1.35), +9/-3, dlogp -0.1223 | 185.3 |
| claude | 7 | score | both | 256 | paired +1.95 (SE 1.41), +9/-4, dlogp -0.1398 | 285.5 |
| claude | 8 | score | both | 256 | paired +0.78 (SE 1.46), +8/-6, dlogp -0.1121 | 266.8 |
| claude | 9 | score | both | 256 | paired +1.56 (SE 1.23), +7/-3, dlogp -0.1597 | 186.0 |
| claude | 10 | score | both | 256 | paired +0.39 (SE 1.17), +5/-4, dlogp -0.1456 | 176.4 |
| ed_a | 1 | score +acts | original | 512 | acc 33.59 | 483.4 |
| ed_a | 2 | score | both | 256 | paired +0.78 (SE 0.78), +3/-1, dlogp -0.0515 | 341.2 |
| ed_a | 3 | score +acts | both | 256 | paired +2.34 (SE 1.35), +9/-3, dlogp +0.1901 | 382.5 |
| ed_a | 4 | score | both | 256 | paired +3.12 (SE 1.55), +12/-4, dlogp +0.2474 | 209.4 |
| ed_a | 5 | score | both | 256 | paired +1.95 (SE 1.61), +11/-6, dlogp +0.1878 | 301.2 |
| ed_a | 6 | score | both | 256 | paired +1.56 (SE 1.56), +10/-6, dlogp +0.1682 | 296.0 |
| ed_a | 7 | score | both | 195 | paired +1.54 (SE 0.88), +3/-0, dlogp -0.0516 | 186.4 |
| ed_a | 8 | score | both | 256 | paired +2.73 (SE 1.6), +12/-5, dlogp +0.1971 | 179.7 |
| ed_a | 9 | score | both | 256 | paired +1.56 (SE 1.83), +13/-9, dlogp +0.1934 | 270.6 |
| ed_a | 10 | score | both | 256 | paired +1.17 (SE 1.61), +10/-7, dlogp +0.1378 | 180.8 |
| ed_b | 1 | score +acts | original | 512 | acc 36.72 | 622.4 |
| ed_b | 2 | score | both | 256 | paired +2.73 (SE 1.51), +11/-4, dlogp +0.2065 | 411.3 |
| ed_b | 3 | score | both | 256 | paired +1.17 (SE 1.17), +6/-3, dlogp +0.1178 | 230.7 |
| ed_b | 4 | score | both | 256 | paired +2.73 (SE 1.79), +14/-7, dlogp +0.2552 | 191.7 |
| ed_b | 5 | score | both | 256 | paired +0.39 (SE 1.52), +8/-7, dlogp +0.2295 | 323.5 |
| ed_b | 6 | score | both | 256 | paired +2.34 (SE 1.91), +15/-9, dlogp +0.2907 | 196.5 |
| ed_b | 7 | score | both | 256 | paired +1.17 (SE 2.03), +15/-12, dlogp +0.2723 | 175.2 |
| ed_b | 8 | score | both | 256 | paired -0.39 (SE 1.52), +7/-8, dlogp +0.1291 | 261.2 |
| ed_b | 9 | score | both | 256 | paired -1.56 (SE 1.1), +2/-6, dlogp +0.1195 | 191.7 |
| ed_b | 10 | score | both | 169 | paired +0.59 (SE 1.03), +2/-1, dlogp -0.1256 | 102.8 |

## Candidates

### claude/equalize_boost

- Tensors: model.layers.22.feed_forward.w2.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 24398

Edit: scale the DIRECT LETTER READOUT (the component of a w2 column that the tied letter rows read through post_ffn_norm and the final norm; a 9-dim subspace) by 0.5 for (a) the 8 letter-specific "leader suppressor" neurons of layer 22's MLP, model.layers.22.feed_forward.w2.weight columns C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310, and (b) 4 early-letter "booster" neurons: L22 column 2212 (A), L23 columns 1685 (A), 2595 (A), 1439 (B). The rest of each column is unchanged. 24,398 weights differ in bf16.
Mechanism: after head L22.H12 moves the chosen option to the answer position, the L22 MLP suppresses the currently leading letter through one neuron per letter (-3 to -4.6 logits when that letter leads). There is no such neuron for A or B, and A/B have boosters. Total suppression of a leader: A -1.2, B -2.2, C..J -4.1 to -5.6. That asymmetry makes the model over-pick early letters (practice: A..C predicted 249 vs 168 true out of 512). Halving the late-letter suppressors and the A/B boosters equalises it.
Evidence (paired, practice): screen256 DEV +2.34 (SE 1.35), +9/-3; dose-response for the suppressors alone: alpha 0.5 +1.56 (SE 0.95), alpha 0 +1.56 (SE 1.23) with more churn; adding layer 23's suppressors hurt (+1.17, SE 1.41). HOLDOUT 512 (never used for choices): parent 32.62 -> 33.98, +1.37 (SE 1.01), +17/-10. Side effect: the correct letter's log-probability falls (-0.13), because the suppressors also damp overconfidence.

### claude/equalize_l22

- Tensors: model.layers.22.feed_forward.w2.weight
- Weights changed (bf16): 16254

Edit: the same as equalize_boost but only part (a): the direct letter readout of the 8 layer-22 late-letter leader suppressors (model.layers.22.feed_forward.w2.weight columns 1812, 464, 7357, 6492, 3344, 397, 2786, 3310) is scaled by 0.5. 16,254 weights differ in bf16.
Evidence: screen256 DEV +1.56 (SE 0.95), +5/-1; HOLDOUT 512: +0.98 (SE 0.85), +12/-7, dlogp(correct) -0.153.

### ed_a/ed_a_v1_letterbias12

- Tensors: model.layers.12.feed_forward.w2.weight, model.layers.15.feed_forward.w2.weight, model.layers.17.feed_forward.w2.weight, model.layers.19.feed_forward.w2.weight, model.layers.20.feed_forward.w2.weight, model.layers.22.feed_forward.w2.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 24269

Removes the answer-letter position prior written by 12 near-constant MLP neurons. Change: for w2 columns (layer,neuron) (20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820), project out (in post_ffn_norm-scaled space) the 9-dim subspace spanned by the centered letter readout directions wn*E[A..J] (column -> (I-QQ^T)(pw*col)/pw); strength 1 (full removal). Mechanism: these neurons are ~constant at the answer position (cv<0.6) and write a fixed preference A,B,C,F >> H,I,J (orig bias in centered logits [1.15 1.66 .57 .16 .03 1.27 -.05 -1.14 -2.09 -1.58]); removal flattens it (S: I -2.59 -> -0.34) with no measurable self-repair (direct-logit prediction matched). Smooth dose-response in direct estimate (peak 0.75-1.0). Measured paired gain: DEV S (DEV[512:768], 256 Qs, not used for selection) acc +2.34 +- 1.35, logp_correct +0.190 +- 0.041; HOLDOUT[0:256] acc +1.17 +- 1.61, logp +0.138 +- 0.039. Subset of ed_a_v2, which measured better.

### ed_a/ed_a_v2_letterbias27h

- Tensors: model.layers.12.feed_forward.w2.weight, model.layers.15.feed_forward.w2.weight, model.layers.17.feed_forward.w2.weight, model.layers.19.feed_forward.w2.weight, model.layers.20.feed_forward.w2.weight, model.layers.21.feed_forward.w2.weight, model.layers.22.feed_forward.w2.weight, model.layers.22.self_attn.o_proj.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 306498

Removes the answer-letter position prior written by near-constant components at the answer position. Change: centered letter-readout subspace (9-dim span of wn*E[A..J] minus mean) projected out (in post-norm-scaled space, strength 1) of: 27 MLP w2 columns (layer,neuron) (20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820),(22,4543),(21,218),(23,6518),(23,4531),(22,5653),(23,514),(23,3073),(23,2189),(21,4481),(22,2212),(22,3998),(23,1439),(23,7466),(21,3455),(21,4902), and head L22h6 (o_proj input cols 768:896). Mechanism: parent over-picks A/B/C/F and starves H/I/J (pred A85 B99 .. I20 J14 vs true A69 B48 .. I41 J49 on 512 DEV); direct logit attribution shows the fixed bias is written by these low-variance, low-content neurons (not by the answer-copy head L22h12). Residual centered bias goes from [1.25 1.34 .99 .21 .14 1.28 -.2 -.88 -2.33 -1.8] to [.3 -.1 -.13 -.13 .01 .78 -.26 -.06 -.04 -.37]; captures ~75% of the offline constant-debias ceiling. Measured paired gain (edited - parent): HOLDOUT (768 practice Qs never used for choosing) acc +1.69 +- 0.96 pp, logp_correct +0.183 +- 0.026; screening set DEV[512:768] acc +3.12 +- 1.55, logp +0.247 +- 0.047; 4-option practice Qs (151) acc 45.7 -> 47.7, A-D bias D -1.31 -> -0.74 (all <=6-option Qs: +1.54 +- 0.88, logp -0.05 +- 0.03).

### ed_b/ed_b_bias16_a10

- Tensors: model.layers.15.feed_forward.w2.weight, model.layers.17.feed_forward.w2.weight, model.layers.19.feed_forward.w2.weight, model.layers.20.feed_forward.w2.weight, model.layers.22.feed_forward.w2.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 32156

alpha = 1.0. Mechanism: the parent over-picks early letters (DEV-512 preds A..J 94 95 78 58 40 54 38 28 15 12 vs gold 59 49 51 68 47 50 42 69 39 38; acc by gold letter A .58 -> I .21). Per-neuron direct logit attribution onto the option-letter unembeddings (through final norm) shows a set of late MLP neurons with near-constant activation whose letter push is an ordinal 'early up / late down' profile (largest: L20 n1000, L22 n190, L17 n2401). Selected by: mean push along the measured offset > 0.08 logit, sd < mean, |help on correct letter| < 0.03 (DEV[:512]). Edit: for each of the 16 neurons, w2[:, j] -= alpha * P w2[:, j], P = projector onto the centered letter-readout subspace (letter unembeddings x final-norm gain x that layer's post_ffn_norm gain); alpha 1 removes exactly the neuron's letter push, other output directions untouched. Tensors: model.layers.{15,17,19,20,22,23}.feed_forward.w2.weight; neurons L15[4408], L17[2401,616], L19[6612], L20[1000,3296,3964,4353], L22[190,1449,2820,4543], L23[1047,4597,1903,3269]. Measured paired gain: DEV[512:768] acc +2.73 +- 1.51, logp +0.207 +- 0.047; HOLDOUT[0:256] acc +0.39 +- 1.52, logp +0.230 +- 0.043; HOLDOUT[256:512] acc -1.56 +- 1.10, logp +0.120 +- 0.045 (holdout 512 pooled: acc -0.59 +- ~0.94, logp +0.175 +- ~0.031). Exact removal (no overshoot) variant, the more conservative choice for 4-option transfer (not measured on 4-option q).

### ed_b/ed_b_bias16_a15

- Tensors: model.layers.15.feed_forward.w2.weight, model.layers.17.feed_forward.w2.weight, model.layers.19.feed_forward.w2.weight, model.layers.20.feed_forward.w2.weight, model.layers.22.feed_forward.w2.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 32337

alpha = 1.5. Mechanism: the parent over-picks early letters (DEV-512 preds A..J 94 95 78 58 40 54 38 28 15 12 vs gold 59 49 51 68 47 50 42 69 39 38; acc by gold letter A .58 -> I .21). Per-neuron direct logit attribution onto the option-letter unembeddings (through final norm) shows a set of late MLP neurons with near-constant activation whose letter push is an ordinal 'early up / late down' profile (largest: L20 n1000, L22 n190, L17 n2401). Selected by: mean push along the measured offset > 0.08 logit, sd < mean, |help on correct letter| < 0.03 (DEV[:512]). Edit: for each of the 16 neurons, w2[:, j] -= alpha * P w2[:, j], P = projector onto the centered letter-readout subspace (letter unembeddings x final-norm gain x that layer's post_ffn_norm gain); alpha 1 removes exactly the neuron's letter push, other output directions untouched. Tensors: model.layers.{15,17,19,20,22,23}.feed_forward.w2.weight; neurons L15[4408], L17[2401,616], L19[6612], L20[1000,3296,3964,4353], L22[190,1449,2820,4543], L23[1047,4597,1903,3269]. Measured paired gain (edited - parent): DEV[512:768] (unseen by selection) acc +2.73 +- 1.79, logp(correct) +0.255 +- 0.070; HOLDOUT[0:256] acc +2.34 +- 1.91, logp +0.291 +- 0.064; HOLDOUT[256:512] acc -0.39 +- 1.52, logp +0.129 +- 0.067 (holdout 512 pooled: acc +0.98 +- ~1.2, logp +0.21). Dose-response on DEV[512:768] (alpha 0.5/1/1.5/2): acc +1.17/+2.73/+2.73/+1.17, logp +.118/+.207/+.255/+.272. Caveat: on 169 practice q with <=5 options (4-option format, where the parent shows no early-letter bias) acc +0.59 +- 1.03 but logp -0.126 +- 0.043 (alpha 1.5 overshoots).

## Selection (hidden questions) and decision

Parent accuracy on the 600 selection questions: **35.0%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| claude/equalize_boost | 35.5 | +0.50 | 0.9 | +0.56 | 16 | 13 | -0.1373 | 24398 |
| claude/equalize_l22 | 35.5 | +0.50 | 0.73 | +0.68 | 11 | 8 | -0.1729 | 16254 |
| ed_a/ed_a_v1_letterbias12 | 34.5 | -0.50 | 0.96 | -0.52 | 15 | 18 | +0.1054 | 24269 |
| ed_a/ed_a_v2_letterbias27h | 34.5 | -0.50 | 1.12 | -0.45 | 21 | 24 | +0.1830 | 306498 |
| ed_b/ed_b_bias16_a10 | 33.67 | -1.33 | 0.94 | -1.41 | 12 | 20 | +0.1182 | 32156 |
| ed_b/ed_b_bias16_a15 | 33.5 | -1.50 | 1.21 | -1.24 | 22 | 31 | +0.1219 | 32337 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

# Report of editor `claude`, generation 1

**What the model gets wrong.**
- Direct-answer accuracy on practice questions is 38% (512 DEV).
- The answer forms very late. Read through the output layer, the residual stream predicts the letter-token prior ("A") until layer 12. It reaches 29% only after layer 22's attention and 38% at the end.
- One head, L22.H12, routes the chosen option to the answer position: +3.1 logits to the correct letter on average.
- When the model is wrong, H12 itself picked the wrong option (its top letter equals the model's wrong pick in 57% of cases). That is a content and knowledge failure that I could not repair.
- The visible, repairable obstacle is a letter-position bias. On 10-option questions the final centred letter logits are A +1.05, B +1.27, C +0.70 ... I -1.53, J -1.45, while the true answers are flat.

**Mechanism.**
- The L22 MLP has one "leader suppressor" neuron per letter, at w2 columns C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310.
- Each fires when its letter leads the stream entering the MLP, pushing that letter down by 3 to 4.6 logits.
- A and B have no such neuron; A even has a booster (L22 n2212), and L23 has A/B boosters (n1685, n2595, n1439).
- Total suppression of the leader, L22 and L23 together, is A -1.2, B -2.2, and C..J -4.1 to -5.6. An early-letter leader keeps its lead and a late-letter leader is knocked down.
- The ceiling of removing the bias entirely, estimated with cross-fitted constant letter offsets on recorded outputs, is about +2 points.

**What I changed.**
- I scaled by 0.5 only the direct letter-readout component of those neurons' w2 columns: the 9-dimensional subspace that the tied letter rows read through post_ffn_norm and the final norm.
- Candidate `equalize_boost` covers the 8 suppressors and the 4 A/B boosters (24,398 weights).
- Candidate `equalize_l22` covers the 8 suppressors only (16,254 weights).

**Evidence.**
- Screening on 256 DEV questions (paired):

  | Edit | Accuracy change | SE |
  |---|---|---|
  | Suppressors, α 0.5 | +1.56 | 0.95 |
  | Suppressors, α 0 | +1.56 | 1.23 |
  | Suppressors + L23 suppressors | +1.17 | 1.41 |
  | **Suppressors + boosters** | **+2.34** | **1.35** |

- HOLDOUT, 512 practice questions never used for any choice:
  - `equalize_boost`: +1.37 (SE 1.01), 17 wrong→right / 10 right→wrong;
  - `equalize_l22`: +0.98 (SE 0.85), 12 / 7.
- Both are consistently positive but not significant on their own.
- Side effect: the correct letter's log-probability falls by 0.13–0.15. The suppressors also damp overconfidence, so a calibration-sensitive metric would get worse.

**What I would try next.**
1. The B over-preference (B predicted 49–58 of 256) is not explained by these neurons. Look for B writers before layer 21. The stream before L22 already carries A +1.18 and B +1.13.
2. Equalise *upward* instead: give A and B a suppressor of the same strength. Many new weights would have to be written, but it would keep calibration.
3. The main error is H12 choosing the wrong option. Check whether its query/key selection is biased by the option's position, using attention patterns from read passes.

### ed_a

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

### ed_b

# ed_b, generation 1: report

## What the model gets wrong
- **Early-letter bias (10-option format).** On DEV[:512] the parent scores 36.7%. Its predicted letters A..J are 94 95 78 58 40 54 38 28 15 12, against gold 59 49 51 68 47 50 42 69 39 38. Accuracy by gold letter falls from .58 (A) to .21 (I). The mean centered final letter logit is A +.98, B +1.51, C +.70 … I -1.82, J -1.55. Removing this constant offset (logit-lens arithmetic, fitted on one half and tested on the other) is worth about +1.5 to +4.7 points and +0.13 to +0.23 log-prob of the correct letter.
- **Not mainly a late-layer loss.** The logit lens shows the answer forms in the last two layers: 27.7% at h22, 35% at h23, 36.9% at h24. Mid layers almost always pick A. Cases where a middle layer has the right answer and the model then loses it are rare (at h22: 26 correct-then-lost versus 72 wrong-then-fixed). Near misses are not dominant either: of the wrong answers, the correct letter is 2nd in 24% and 3rd in 19%.
- **Retrieval circuit.** Layer 22 attention head 12 is the answer-retrieval head: correct-letter direct logit attribution (DLA) +3.03, 34% accuracy on its own. Head 15 also helps (+0.48). Heads 11, 13 and 14 push against the correct letter. MLP22 suppresses whatever attn22 retrieved: the per-question correlation of their letter pushes is a median of -0.88. It has one suppressor neuron per letter (C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310), and I found none for A or B.
- **Remaining errors are upstream.** On questions still wrong after my edit, the winning wrong letter's margin comes from head 12 retrieving the wrong option (+4.97 on average, positive in 91% of cases). MLP22 works against it (-2.30). Head 12 is correct on only 6% of these questions. The remaining errors are the head attending to the wrong option, not late suppression.
- **Rejected ideas (lens screening).** Removing the MLP22 suppressors, removing the anti-correct L22 heads, boosting heads 12 or 15, and equalizing head 12's per-letter gain all cost 0.2 to 2 nats of log-prob.

## What I changed (both candidates use the same mechanism)
- **Selection.** Per-neuron DLA on MLP layers 14–23 (code/007, code/008) found late MLP neurons with near-constant activation whose letter push has an ordinal "early up, late down" profile. The largest are L20 n1000, L22 n190 and L17 n2401. I selected 16 neurons where the mean push along the offset is above 0.08, its sd is below its mean, and the push on the correct letter is below 0.03 in absolute value. Selection used DEV[:512] only:
  - L15: 4408
  - L17: 2401, 616
  - L19: 6612
  - L20: 1000, 3296, 3964, 4353
  - L22: 190, 1449, 2820, 4543
  - L23: 1047, 4597, 1903, 3269
- **Edit.** For each neuron, `w2[:, j] -= alpha * P w2[:, j]`. P projects onto the centered option-letter readout subspace: the letter unembeddings × the final-norm gain × that layer's post_ffn_norm gain. At alpha = 1 this removes exactly the neuron's letter push and leaves its other output directions untouched. Tensors: `model.layers.{15,17,19,20,22,23}.feed_forward.w2.weight`, about 32k weights.
- **Lens check.** On DEV-512 the lens predicted +0.19 log-prob for this edit, and the real network gave +0.21. The removed component is largely linear in its effect on the answer.
- After the edit the monotone gradient is gone. Accuracy for gold A–E versus F–J goes from .42/.26 to .37/.36 on HOLD[0:256] at alpha 1.5. Residual A and F preferences remain because they were not targeted.

## Evidence (paired edited minus parent; questions never used for selection)
| alpha | set | acc diff | logp diff |
|---|---|---|---|
| 0.5 | DEV[512:768] | +1.17 ± 1.17 | +0.118 ± 0.024 |
| 1.0 | DEV[512:768] | +2.73 ± 1.51 | +0.207 ± 0.047 |
| 1.5 | DEV[512:768] | +2.73 ± 1.79 | +0.255 ± 0.070 |
| 2.0 | DEV[512:768] | +1.17 ± 2.03 | +0.272 ± 0.092 |
| 1.0 | HOLD[0:256] | +0.39 ± 1.52 | +0.230 ± 0.043 |
| 1.5 | HOLD[0:256] | +2.34 ± 1.91 | +0.291 ± 0.064 |
| 1.0 | HOLD[256:512] | -1.56 ± 1.10 | +0.120 ± 0.045 |
| 1.5 | HOLD[256:512] | -0.39 ± 1.52 | +0.129 ± 0.067 |
| 1.5 | 169 q with ≤5 options | +0.59 ± 1.03 | **-0.126 ± 0.043** |

- **Pooled holdout (512 questions).** Alpha 1.5: accuracy +0.98 ± ~1.2, log-prob +0.21. Alpha 1.0: accuracy -0.59 ± ~0.94, log-prob +0.175.
- **The log-prob gain is robust; the accuracy gain is not significant.** On HOLD[256:512] the parent already had no late-letter disadvantage (gold A–E .38 versus F–J .41), so the edit only traded letters there.
- **In the 4-option format the parent shows no early-letter bias.** The alpha 1.5 overshoot costs log-prob there, which is a GPQA transfer risk.

## Candidates
1. `ed_b_bias16_a15`: alpha 1.5. Best measured holdout gain.
2. `ed_b_bias16_a10`: alpha 1.0. Exact removal with no overshoot; more conservative for 4-option transfer.

## What I would try next
- **Make the debias format-aware.** The bias is specific to long option lists (H–J under-picked) and absent with 4 options. Look for what makes these neurons' letter push depend on the number of options, for example attention to the option count, rather than removing it uniformly.
- **Target the residual A and F preferences,** and selection neurons below layer 14. I recorded no MLP activations there (mlp13 and h12 carry about +0.8 of the offset).
- **The real lever is head 12 of L22, which picks the wrong option.** Study its QK circuit with read passes that return its attention patterns and the values at the option-letter positions. That would show whether it is misled by option position or option length, or whether the upstream "belief" is wrong.
- **Spend more passes on holdout.** At n = 256 the accuracy SE (1.1 to 1.9) is as large as the effect.

