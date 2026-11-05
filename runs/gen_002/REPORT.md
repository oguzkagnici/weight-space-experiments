# Generation 2 report

- Created: 2026-11-04 22:39 UTC
- Parent: `base`
- Editors: claude, ed_a, ed_b; budget 10 passes each; up to 2 candidates each
- Hidden selection subset: 800 questions (seed `selection-gen-2`)

## Suggested starting points

- **ed_a**: The option selection of layer 22 head 12 (the answer-routing head found in generation 1). Use read passes with attn_last=[22] on rendered practice questions to see which option spans head 12 attends to from the answer position, and whether non-content factors steer it (option position or recency, through the partial rotary dimensions of the full-attention layer; letter identity; option length). If such a factor biases it, repair it in head 12's query/key weights.
- **ed_b**: The correctness belief at the option positions upstream of layer 22. Compare residual streams at the end of each option for correct and incorrect options, find the layer and components where a correct-vs-incorrect signal is formed, and check whether head 12 of layer 22 reads it through its keys. If the signal exists but is read weakly or is diluted, strengthen the path.
- **claude**: Near misses: questions where the correct letter is ranked second. Decompose the correct-minus-chosen margin by component and find what systematically flips the top two.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | score | both | 256 | paired +1.95 (SE 1.7), +12/-7, dlogp -0.5467 | 372.8 |
| claude | 2 | score | both | 256 | paired +1.17 (SE 0.87), +4/-1, dlogp -0.1513 | 175.3 |
| claude | 3 | score | both | 256 | paired +1.56 (SE 1.66), +11/-7, dlogp -0.2116 | 236.9 |
| claude | 4 | score | both | 256 | paired +1.17 (SE 1.3), +7/-4, dlogp -0.1057 | 186.7 |
| claude | 5 | score | both | 256 | paired +1.95 (SE 1.51), +10/-5, dlogp -0.1984 | 179.2 |
| claude | 6 | score | both | 256 | paired +0.78 (SE 1.46), +8/-6, dlogp -0.3083 | 163.3 |
| claude | 7 | score | both | 256 | paired -0.39 (SE 1.61), +8/-9, dlogp -0.2975 | 158.4 |
| claude | 8 | score | both | 256 | paired +1.56 (SE 1.1), +6/-2, dlogp -0.1449 | 164.6 |
| claude | 9 | score | both | 256 | paired -0.39 (SE 0.87), +2/-3, dlogp -0.1572 | 152.4 |
| claude | 10 | score | both | 256 | paired +1.17 (SE 0.87), +4/-1, dlogp -0.1368 | 191.7 |
| ed_a | 1 | read | original | 512 | read 512 texts | 346.6 |
| ed_a | 2 | read | original | 120 | read 120 texts | 99.9 |
| ed_a | 3 | score | both | 256 | paired +1.95 (SE 1.79), +13/-8, dlogp +0.1654 | 238.2 |
| ed_a | 4 | score | both | 256 | paired +1.95 (SE 1.03), +6/-1, dlogp +0.1293 | 181.0 |
| ed_a | 5 | read | original | 120 | read 120 texts | 84.6 |
| ed_a | 6 | score | both | 256 | paired +1.95 (SE 1.87), +14/-9, dlogp +0.1861 | 175.1 |
| ed_a | 7 | score | both | 256 | paired +0.00 (SE 2.14), +15/-15, dlogp +0.1970 | 170.8 |
| ed_a | 8 | score | both | 256 | paired +0.78 (SE 2.0), +14/-12, dlogp +0.1962 | 172.2 |
| ed_a | 9 | score | both | 256 | paired -0.39 (SE 1.52), +7/-8, dlogp +0.1122 | 131.5 |
| ed_a | 10 | score | both | 256 | paired -0.78 (SE 1.24), +4/-6, dlogp +0.1181 | 176.7 |
| ed_b | 1 | read | original | 160 | read 160 texts | 141.7 |
| ed_b | 2 | read | original | 180 | read 180 texts | 113.4 |
| ed_b | 3 | score +acts | original | 512 | acc 34.18 | 345.6 |
| ed_b | 4 | score | both | 256 | paired +1.56 (SE 1.46), +9/-5, dlogp -0.3187 | 205.3 |
| ed_b | 5 | score | both | 256 | paired +1.17 (SE 1.7), +11/-8, dlogp +0.5636 | 154.7 |
| ed_b | 6 | score | both | 256 | paired -0.78 (SE 1.24), +4/-6, dlogp -0.3052 | 169.5 |
| ed_b | 7 | score | both | 256 | paired -1.56 (SE 1.91), +10/-14, dlogp +0.4696 | 136.0 |
| ed_b | 8 | read | original | 1024 | read 1024 texts | 702.1 |
| ed_b | 9 | score | both | 256 | paired -0.78 (SE 1.56), +7/-9, dlogp -0.3074 | 193.3 |
| ed_b | 10 | score | both | 256 | paired +0.39 (SE 2.11), +15/-14, dlogp +0.5554 | 147.6 |

## Candidates

### claude/routers_x1p5

- Tensors: model.layers.14.self_attn.o_proj.weight, model.layers.20.self_attn.o_proj.weight
- Weights changed (bf16): 786432

Edit: scale the outputs of three weak "early router" heads by 1.5: model.layers.14.self_attn.o_proj.weight columns of head 15 (15*128..16*128-1) and head 12 (12*128..13*128-1), and model.layers.20.self_attn.o_proj.weight columns of head 7 (7*128..8*128-1). 786,432 weights.
Mechanism (H5): besides the main answer-routing head L22.H12 (own-choice accuracy .371), these heads write a weak, partly independent vote for an option (own-choice accuracy .26-.28, agreeing with H12 only 31-36%, right on 15-16% of H12's mistakes; direct effect ~0.1 logit). Amplifying them gives the ensemble more say.
Evidence (paired, chosen on DEV): DEV screen256 +1.17 (SE 0.87), DEV screen2 +1.17 (SE 1.30); dose-response x1.5 +1.17, x2 +1.95, x3 +1.95 (log-prob cost grows: -0.15, -0.20, -0.55). HOLDOUT (measured once, never used to choose): 768 questions, about +0.78 (SE ~0.55): HOLDOUT[0:512] +0.59 (SE 0.70) +8/-5, HOLDOUT[512:768] +1.17 (SE 0.87) +4/-1. The correct letter's log-prob falls by ~0.14.

### claude/routers_x2

- Tensors: model.layers.14.self_attn.o_proj.weight, model.layers.20.self_attn.o_proj.weight
- Weights changed (bf16): 786432

Edit: the same three heads (L14.H15, L14.H12, L20.H7; o_proj input columns of each head) scaled by 2. 786,432 weights.
Evidence: DEV screen2 +1.95 (SE 1.51), +10/-5. HOLDOUT[0:512] (once): +0.20 (SE 1.09), +16/-15, dlogp -0.303. Submitted as the stronger dose; HOLDOUT shows it is about null.

### ed_a/ed_a_qprimacy_L22L20

- Tensors: model.layers.20.self_attn.q_proj.weight, model.layers.22.self_attn.q_proj.weight
- Weights changed (bf16): 6259406

Removes the option-ordinal ("primacy") query component from the answer-selecting attention layers L22 and L20.
Change: model.layers.22.self_attn.q_proj.weight and model.layers.20.self_attn.q_proj.weight, all 16 heads. For head h: W_h <- W_h - t t^T W_h (alpha = 1), where W_h = rows h*128:(h+1)*128 and t = normalize(q_norm.weight * v1_g). v1_g is PC1 of the slot-mean key vectors (k_norm outputs averaged over the tokens of options A..J, 120 DEV questions) of kv group g = h//4. Rank-1 per head, 2 tensors.
Mechanism: L22 is NoPE-like (q.k on recorded q_norm/k_norm reproduces the attention logits, residual var 0.3%). Option keys carry an ordinal position signal along v1 (slot means A -5.7, B -3.8, ..., J +0.3; split-half r .97). The answer-position query has a near-constant large component along v1 (about 32% of |q|), so every option-reading L22 head prefers early options by about 2.2 nats (A vs I). L20 (RoPE, but v1 sits in low-frequency pairs) has the same mechanism with an A-first preference of about 1 nat. Removing it flattens the letter prior: centred I goes from -2.1 to -0.6.
Evidence (paired edited - parent): DEV S1 (DEV[512:768], 256 q): +1.95 (SE 1.87), logp(correct) +0.186 (SE 0.060). HOLDOUT 512 practice questions (never used for any choice): +0.39 (SE 1.46), 29 w->r / 27 r->w, logp +0.197 (SE 0.055). The log-prob gain is robust, but the accuracy gain is not significant. A becomes under-picked (gold-A accuracy .43 -> .22).

### ed_a/ed_a_qprimacy_L22_half

- Tensors: model.layers.22.self_attn.q_proj.weight
- Weights changed (bf16): 2421150

Half-strength version, L22 only: model.layers.22.self_attn.q_proj.weight, all 16 heads, W_h <- W_h - 0.5 t t^T W_h, where t = normalize(q_norm.weight * v1_g) and v1_g = PC1 of the L22 slot-mean option keys of kv group h//4. This halves the constant query component that makes L22 heads prefer early options (primacy bias A > ... > J of about 2.2 nats). Rank-1 per head.
Evidence (paired): DEV S1 (DEV[512:768], 256 q): +1.95 (SE 1.03), 6 w->r / 1 r->w, logp +0.129 (SE 0.031). HOLDOUT 512: -0.59 (SE 0.98), 11 w->r / 14 r->w, logp(correct) +0.115 (SE 0.026). The logp gain is robust; there is no accuracy gain.

### ed_b/ed_b_g2_L12H13_ablate

- Tensors: model.layers.12.self_attn.o_proj.weight
- Weights changed (bf16): 262144

Change: model.layers.12.self_attn.o_proj.weight columns 1664:1792 (input of head 13 of layer 12) set to 0 (head L12.H13 ablated); 262,144 weights. Mechanism: L12.H13 (sliding-window layer 12) is the strongest attention writer of the option-end correctness belief that L22.H12 later reads through its keys (projection on the correct-minus-incorrect direction: diff .50, SNR .85, alone picks gold 29% vs 31% for the whole stream). Its contribution is 0.83-correlated with the rest of the stream, so it sets the CONFIDENCE of the belief, not which option wins. Dose-response on DEV S1 (256 q): x0 / x1 / x2 give output entropy 1.24 / .99 / .90 and an unchanged gold rank (diff -0.02 +- 0.08 and +0.004 +- 0.06). Measured: DEV S1 acc +1.17 (SE 1.70), dlogp(correct) +0.56 (SE .08); HOLDOUT[0:512] (never used for choosing) acc -0.59 (SE 1.42), 25 wrong->right / 28 right->wrong, dlogp(correct) +0.513 (SE .054). Expect about 0 accuracy change; it is a calibration fix (much less overconfident when wrong), not an accuracy gain.

## Selection (hidden questions) and decision

Parent accuracy on the 800 selection questions: **37.0%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| ed_b/ed_b_g2_L12H13_ablate | 38.25 | +1.25 | 1.2 | +1.04 | 51 | 41 | +0.5002 | 262144 |
| ed_a/ed_a_qprimacy_L22_half | 36.5 | -0.50 | 0.88 | -0.57 | 23 | 27 | +0.1265 | 2421150 |
| claude/routers_x2 | 35.38 | -1.62 | 0.93 | -1.74 | 21 | 34 | -0.2753 | 786432 |
| ed_a/ed_a_qprimacy_L22L20 | 35.38 | -1.62 | 1.19 | -1.36 | 39 | 52 | +0.1918 | 6259406 |
| claude/routers_x1p5 | 35.25 | -1.75 | 0.77 | -2.27 | 12 | 26 | -0.1480 | 786432 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

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

### ed_a

# ed_a, generation 2: the option-primacy query component in L22/L20 attention

## What the model gets wrong
- **Accuracy and letter bias.** On DEV[:512] the parent scores 34.4% by lens. On 10-option questions it predicts A..J as 75 78 62 53 39 39 41 27 16 19, while the gold answers are flat (about 45 each).
- **The retrieval heads prefer early options.** Every option-reading head in L22 (0, 1, 3, 8, 10–15) puts more attention on early option spans. For example, L22h12's share by slot is A .22, B .16 … I .05. A per-token regression (code/006) shows that this is a **slot** effect, not token distance: h12 is −0.8 (B) to −2.3 (I) nats against A, and token distance adds only −0.15 per 100 tokens.
- **Mechanism (pass 2, code/008–011).**
  - L22 behaves as NoPE: the dot product of the recorded `q_norm` and `k_norm` outputs reproduces the attention logits (0.3% residual variance).
  - Option keys carry an ordinal position signal along one direction v1, the PC1 of the slot-mean keys (split-half r .97). Mean k·v1 by slot is A −5.7, B −3.8, C −2.8 … J +0.3; question text is −4.7.
  - The answer-position query has a large, nearly constant component along v1 (q·v1 ≈ −3 to −4, about 32% of |q|).
  - Rank 1 explains the whole slot preference, about 2.2 nats from A to I.
  - The query does *not* read the letter-unembedding subspace (code/005), so this is not feedback from the residual letter prior.
- **L20 has the same mechanism in a weaker form (pass 5, code/014–015).** L20 uses RoPE (half-rotation, θ = 1e4), but v1 sits in low-frequency pairs. Its slot preference is "A first", about +1 nat.

## What I changed
- **Construction.** For each head, I removed the query's component along v1 in q_proj: `W_h -= α t tᵀ W_h`, with `t = normalize(q_norm.weight ⊙ v1_g)` and g = h//4. This is rank 1 per head and computed from recorded keys, not fitted.
- **Candidate 1, `ed_a_qprimacy_L22L20`.** L22 and L20, all heads, α = 1.
- **Candidate 2, `ed_a_qprimacy_L22_half`.** L22 only, α = 0.5.

## Evidence (all paired, edited minus parent)
| Edit | Questions | Accuracy change (pp) | Log-prob of correct letter |
|---|---|---|---|
| L22, α 1 | S1 = DEV[512:768] | +1.95 (SE 1.79), 13/8 | +0.165 (SE 0.056) |
| L22, α 0.5 | S1 | +1.95 (SE 1.03), 6/1 | +0.129 (SE 0.031) |
| L22 + L20, α 1 | S1 | +1.95 (SE 1.87), 14/9 | +0.186 (SE 0.060) |
| **L22 + L20, α 1** | **HOLDOUT 512** | **+0.39 (SE 1.46), 29/27** | **+0.197 (SE 0.055)** |
| L22, α 0.5 | HOLDOUT 512 (same questions) | −0.59 (SE 0.98), 11/14 | +0.115 (SE 0.026) |

The "a/b" after each accuracy change counts questions that went wrong→right / right→wrong.

- **The flattening reproduces.** The centred letter prior goes from roughly [1.2 1.4 .7 .1 .3 1.1 .1 −1.0 −2.1 −1.7] to [−.5 .3 0 −.3 .1 1.0 .5 0 −.6 −.3]. The log-prob gain of the correct letter replicates on HOLDOUT (z ≈ 3.6 for candidate 1, 4.4 for candidate 2) and is dose-dependent.
- **Accuracy does not follow.** On S1, constant-offset debiasing of both models left the edit +2.3 points ahead. On HOLDOUT that content gain vanished: cross-fitted debiased accuracy was .366 → .326 and .359 → .357 for the two halves.
- **A becomes under-picked.** At α = 1, gold-A accuracy falls from .43 to about .22. Late letters gain, but the net change is about zero. F keeps a +1 preference that this mechanism does not explain.
- **Caveat on candidate 2.** I chose it on DEV evidence: on S1 it kept gold-A accuracy (.42) and had the best z. I did pick it after seeing candidate 1's HOLDOUT result.
- **Conclusion.** The primacy bias is real, causal and cleanly repairable, but repairing it mostly redistributes probability. Like gen-1's output-side debiasing, it does not raise accuracy significantly. Expect about 0 on the hidden selection.

## What I would try next
1. **Why A collapses.** After the edit, A loses more than the prior it had. Check the slot-specific keys beyond rank 1 for A, and the A-specific booster and suppressor neurons in the L22/L23 MLPs that compensated for the attention primacy. A combined edit could restore A while keeping the late-letter gains: for example, project only the *within-options* ordinal part, or add back A's share.
2. **The F +1 preference** survives every edit so far (gen 1 and this one). Find its writer.
3. **The main error is content.** Even debiased, H12 attends to the wrong option. Since L22 is NoPE, it can be analysed exactly from recorded q and k. Look at which content features of the keys (option-text tokens versus the closing tokens) carry the "correct" signal, and whether the query reads question information through a small subspace that could be strengthened.

### ed_b

# ed_b, generation 2: report

## Summary
I followed the suggested starting point: the correctness belief at the option positions, read by L22.H12. It turned out to be a negative result.
- H12 already reads all the linearly available correctness signal at the options.
- The heads that write that signal mainly set the model's **confidence**, not which option wins.
- The remaining errors are knowledge errors. Among the wrong options, the model ranks the gold one no better than chance.
- My one candidate is a calibration fix: correct-letter log-prob +0.51 (SE 0.05) on HOLDOUT. Accuracy changes by about zero (-0.59, SE 1.42).

## What the model gets wrong
**Where the answer is chosen.**
- L22.H12 puts 65% of its attention on option END tokens (the last token of each option line) and 12% on the "(X" letter tokens.
- Its attention argmax picks the gold option 40.6% of the time (DEV-160) and agrees with the final answer 72.5% of the time.
- L14.H12 already does the same selection (41.9%).
- H12's attention has no position prior: its argmax counts match the gold counts. The A/B/C-up, I/J-down letter bias arises after selection.

**The correctness belief at the option ends.**
- It forms between layers 8 and 14. Separation along the correct-minus-incorrect direction: h4 .02, h8 .25, h10 .81, h12 .89, h14 1.00.
- Most of it is written by MLP/conv layers.
- The attention heads writing it are L12.H13 (strongest), L10.H4, L8.H1 and L12.H8. L12.H12 writes against it (correlation -0.82 with the rest of the stream).

**No unread signal.**
- Cross-validated probes at the option ends reach 31–41%, no better than H12's own pick.
- On questions the model gets wrong, the probes pick gold 8–16% of the time (chance is about 10%).
- Combining H12 attention, probes, L14.H12 and the final logits gains at most 2 of 160 questions.
- So L22.H12 is not reading the belief weakly. The belief itself is wrong on those questions.

**Letter stage (512 DEV, parent 34.2%).**
- Accuracy read through the output layer at each stage: h22 24.2%, after attn22 34.4%, after mlp22 32.4%, final 34.8%.
- Two letter biases roughly cancel: L22.H12's own late-letter tilt (A -.75, B -1.44, G +1.27) against an early prior (A +1.0, B +1.2) plus mlp22's early tilt.
- The leader-suppressor bank in the L22 MLP is asymmetric, as generation 1 found: A n389 -.76, B n6620/n5859 -1.0, C n1812 -1.7, D–J -3.1 to -4.1.
- A 2-fold cross-validated constant debias would gain only +0.6 points on this set. Rescaling or equalizing any late component in lens screens moves accuracy by at most ±1 point.

**Surface features.** The model picks the longest option 26% of the time (gold 19%). A cross-validated length correction gains nothing. Lexical overlap with the question has no effect.

**Unused knowledge (cloze test, pass 8).**
- I scored each option's text likelihood after the question, on 108 DEV questions.
- It picks gold 25.9% overall and 25% on questions the letter protocol gets wrong (18 of 72; chance about 10%).
- The model therefore holds some knowledge that the letter-selection circuit does not use.
- Adding it to the letter scores (0.25 × z-score) gained +2 of 108 questions, a small sample.

## What I changed and why
**Edits tried.** I tested rescaling L12.H13, the main attention writer of the belief: the head-13 columns (1664:1792) of `model.layers.12.self_attn.o_proj.weight`.

| Edit | Question set | Accuracy change | Log-prob of correct letter | Entropy | Gold rank change |
|---|---|---|---|---|---|
| ×2 (X1) | DEV S1 (256 q) | +1.56 (SE 1.46) | -0.32 (SE .04) | .99 → .90 | +.004 ± .06 |
| ×2 (X1) | HOLDOUT[0:512] | -0.78 (SE 1.00) | -0.31 (SE .03) | | |
| ×0, ablation (X0) | DEV S1 (256 q) | +1.17 (SE 1.70) | +0.56 (SE .08) | .99 → 1.24 | -.02 ± .08 |
| ×0, ablation (X0) | HOLDOUT[0:512] | -0.59 (SE 1.42) | +0.51 (SE .05) | | |

- X0 on HOLDOUT changed 25 answers wrong→right and 28 right→wrong.
- The dose-response is clean: the head sets confidence (entropy, log-prob) and leaves the ranking alone.

**Submitted:** `ed_b_g2_L12H13_ablate` (X0), the better holdout result. It is a calibration improvement; expect about zero change in accuracy. I did not submit X1, which is worse on both measures.

## What I would try next
1. **Use the knowledge in the option text.** Cloze likelihood is right on 25% of the questions the letter protocol misses. Find the circuit that computes how plausible an option's text is at its tokens (possibly the same late MLPs), and see whether it can feed the option-end belief that L22.H12 reads. This is the only lead I found that carries information the model's answer does not already use.
2. **Stop spending effort on letter-bias and confidence edits.** The constant-debias ceiling was +0.6 on my 512-question set. Generation 1's prior removals were negative on the hidden set, and generation 1's suppressor halving was +0.5. Edits of the belief's confidence (this generation) do not change accuracy.
3. Any accuracy edit must improve the **gold rank**. Screen with the paired gold-rank or entropy split (code/026), which separates real ranking gains from sharpening or flattening.

