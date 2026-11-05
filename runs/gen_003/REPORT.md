# Generation 3 report

- Created: 2026-11-05 01:14 UTC
- Parent: `base`
- Editors: claude, ed_a, ed_b; budget 10 passes each; up to 2 candidates each
- Hidden selection subset: 800 questions (seed `selection-gen-3`)

## Suggested starting points

- **ed_a**: Unused option-text knowledge. In generation 2, the likelihood of each option text after the question (a cloze score) picked the gold option on 25% of the questions the letter protocol gets wrong. Find where the model represents an option text as plausible at the option tokens, compare that with the option-end correctness belief that layer 22 head 12 reads, and look for a targeted edit that routes the missing knowledge into that belief. Screen by paired gold rank, not only accuracy.
- **ed_b**: The content side of layer 22 head 12. Layer 22 has no positional rotation, so its attention scores are exact query-key dot products. Find which key features at option tokens carry the correct-option signal, and which question features the answer-position query uses to match them. Then strengthen the content match in head 12 query/key weights without reintroducing letter or position bias. Screen by paired gold rank.
- **claude**: The belief-forming writers in layers 8-14 (MLP, conv and attention outputs at the option-end tokens). Find components whose contribution separates correct from incorrect options in a way that generalises (cross-validated), and strengthen them. Screen by paired gold rank, not by confidence.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 192 | read 192 texts | 164.5 |
| claude | 2 | score | both | 256 | paired -2.34 (SE 2.07), +11/-17, dlogp +0.4200 | 175.6 |
| claude | 3 | score | both | 256 | paired -0.39 (SE 2.03), +13/-14, dlogp +0.3727 | 179.9 |
| claude | 4 | score | both | 256 | paired +0.78 (SE 1.24), +6/-4, dlogp +0.1922 | 163.9 |
| claude | 5 | score | both | 256 | paired +0.00 (SE 1.24), +5/-5, dlogp +0.1867 | 168.4 |
| claude | 6 | score | both | 256 | paired +1.17 (SE 1.03), +5/-2, dlogp +0.2329 | 208.8 |
| claude | 7 | score | both | 256 | paired +1.17 (SE 1.17), +6/-3, dlogp +0.2774 | 210.6 |
| claude | 8 | score | both | 256 | paired -0.78 (SE 2.0), +12/-14, dlogp +0.4675 | 190.6 |
| claude | 9 | score | both | 256 | paired +0.39 (SE 1.96), +13/-12, dlogp +0.6019 | 185.1 |
| claude | 10 | score | both | 256 | paired -1.17 (SE 1.17), +3/-6, dlogp +0.2757 | 182.8 |
| ed_a | 1 | read | original | 946 | read 946 texts | 785.0 |
| ed_a | 2 | read | original | 400 | read 400 texts | 297.2 |
| ed_a | 3 | score | both | 256 | paired -0.39 (SE 1.3), +5/-6, dlogp +0.1278 | 346.4 |
| ed_a | 4 | score | both | 256 | paired -0.78 (SE 1.24), +4/-6, dlogp -0.2760 | 177.2 |
| ed_a | 5 | score | both | 256 | paired +0.39 (SE 1.3), +6/-5, dlogp +0.1358 | 183.4 |
| ed_a | 6 | score | both | 256 | paired +1.17 (SE 1.3), +7/-4, dlogp +0.1266 | 195.1 |
| ed_a | 7 | score | both | 256 | paired +1.17 (SE 1.41), +8/-5, dlogp +0.1441 | 195.0 |
| ed_a | 8 | score | both | 256 | paired +0.00 (SE 1.36), +6/-6, dlogp +0.1046 | 195.3 |
| ed_a | 9 | score | both | 232 | paired +0.86 (SE 1.06), +4/-2, dlogp +0.1598 | 166.8 |
| ed_a | 10 | read | original | 1016 | read 1016 texts | 451.4 |
| ed_b | 1 | read | original | 240 | read 240 texts | 228.5 |
| ed_b | 2 | score +acts | original | 512 | acc 35.16 | 542.7 |
| ed_b | 3 | read | original | 120 | read 120 texts | 112.5 |
| ed_b | 4 | score | both | 256 | paired +1.17 (SE 0.87), +4/-1, dlogp +0.0222 | 201.9 |
| ed_b | 5 | score | both | 256 | paired +0.39 (SE 1.17), +5/-4, dlogp +0.0404 | 205.5 |
| ed_b | 6 | score | both | 256 | paired +0.00 (SE 0.96), +3/-3, dlogp +0.0653 | 189.3 |
| ed_b | 7 | score | both | 256 | paired -0.78 (SE 1.11), +3/-5, dlogp +0.0455 | 183.4 |
| ed_b | 8 | score | both | 256 | paired +0.78 (SE 0.96), +4/-2, dlogp -0.0059 | 186.1 |
| ed_b | 9 | score | both | 256 | paired +0.00 (SE 1.11), +4/-4, dlogp +0.0704 | 214.8 |
| ed_b | 10 | score | both | 256 | paired +0.00 (SE 0.96), +3/-3, dlogp +0.0240 | 155.4 |

## Candidates

### claude/l12h13_ablate_retest

- Tensors: model.layers.12.self_attn.o_proj.weight
- Weights changed (bf16): 262144

Edit: ablate L12.H13 (its 128 o_proj input columns set to 0), rebuilt from the published description of generation-2 candidate ed_b/ed_b_g2_L12H13_ablate. 262,144 weights.
Purpose: an unbiased re-test, on a fresh hidden subset, of the best hidden candidate of generation 2 (there +1.25, SE 1.20, z 1.04; gold rank -0.11; dlogp +0.50), whose estimate is inflated by being the best of five.
My evidence: DEV 512 -1.37 (SE 1.45), gold rank +0.076 (SE 0.057), dlogp +0.40; HOLDOUT 512 -0.20 (SE 1.40), +25/-26, gold rank -0.104 (SE 0.058), dlogp +0.535 (SE 0.054).

### claude/l12h13_half

- Tensors: model.layers.12.self_attn.o_proj.weight
- Weights changed (bf16): 262144

Edit: scale the output of attention head L12.H13 by 0.5 (input columns 13*128..14*128-1 of model.layers.12.self_attn.o_proj.weight). 262,144 weights.
Mechanism: L12.H13 is the main attention writer of the option-correctness belief at the option-end tokens (generation 2, ed_b); its strength sets the CONFIDENCE of the final answer more than its ranking. Halving it is a dose the generation-2 search never tested (x0 and x2 were tested).
Evidence (paired; chosen on DEV): DEV 512 +0.39 (SE 0.87), gold rank -0.002 (SE 0.027), dlogp +0.19. HOLDOUT 768 (measured once per question, never used for choosing): +0.39 (SE 0.65), +14/-11; gold rank -0.048 (SE 0.026); dlogp(correct) +0.262 (SE 0.020). Expect ~0 accuracy and a robust calibration gain.

### ed_a/ed_a_g3_prior17

- Tensors: model.layers.22.feed_forward.w2.weight, model.layers.23.feed_forward.w2.weight
- Weights changed (bf16): 34233

Removes the content-free answer-letter prior written by 17 near-constant late MLP neurons. Change: model.layers.22.feed_forward.w2.weight columns 190, 1449, 2820, 4543, 5653, 2212, 3998 and model.layers.23.feed_forward.w2.weight columns 1047, 4597, 1903, 6518, 4531, 514, 3073, 2189, 1439, 7466: in post_ffn_norm-scaled space (u = g_post * col) the component in the 9-dim centered letter-readout subspace (span of embedding_norm.weight * E[A..J] minus their mean) is projected out at strength 1, col <- (u - QQ^T u)/g_post; the rest of each column is unchanged (34,233 weights differ in bf16). Mechanism: these neurons fire nearly constantly at the answer position and write a fixed A/B/C/F-up, H/I/J-down letter preference (gen-1 ed_a list, L22/L23 subset); the parent predicts A..J 174 164 162 119 96 99 73 50 34 29 vs gold 134 104 115 115 74 95 83 101 95 84 on HOLDOUT. Removing their letter write flattens this without self-repair (DLA screen predicted logp +0.14, measured +0.13). Evidence (paired edited - parent): DEV2[0:256] -0.39 (SE 1.30), DEV pass1[0:256] +0.39 (SE 1.30); HOLDOUT[0:1000], measured once after choosing on DEV: +0.80 (SE 0.65, z 1.23), 25 wrong->right / 17 right->wrong, logp(correct) +0.133 (SE 0.0125), gold rank -0.08. Gains come from gold H/I/J (+4-6 pts each) at a cost on gold B (-7). Expect accuracy ~+0.5-1 (not significant alone); calibration gain is robust.

### ed_b/letterproj_L22_all

- Tensors: model.layers.22.self_attn.q_proj.weight
- Weights changed (bf16): 2964771

Removes the content-free 'option letter token' vote from all layer-22 attention queries. Change: model.layers.22.self_attn.q_proj.weight only; for every head h (16 heads, rows h*128:(h+1)*128): W_h <- W_h - a w_g w_g^T W_h, rank 1 per head, a = 1.0 (head 12: a = 1.5), w_g = normalize(q_norm.weight * u_g), u_g = mean over 240 DEV questions of (mean k_norm key at the option letter tokens '(X' minus mean key at the option end and newline tokens) for kv group g = h//4. Mechanism: L22 is NoPE (q.k reproduces attention); its answer-copy head H12 puts 12% of its attention on the letter tokens '(X', whose keys precede the option text (causal) and so carry only a slot code, not the option's content; many other L22 heads do the same (7-50%). Lens (DEV, recorded attention): dropping letter tokens raises H12's option pick +5/-2 and that of H0/H3/H11 by +8 to +20 net; q_norm renormalizes the query after the projection. Evidence (paired, edited - parent): DEV3 256 q: acc +0.39 (SE 1.17), 5/4, dlogp(correct) +0.040 (SE 0.023), gold rank -0.078 (SE 0.039). HOLDOUT 512 q (measured once, after choosing on DEV): acc 0.00 (SE ~0.74), 7 wrong->right / 7 right->wrong, dlogp(correct) +0.047 (SE ~0.017), gold rank -0.031 (SE ~0.027). Accuracy-neutral; small replicated calibration gain.

### ed_b/letterproj_L22_h12

- Tensors: model.layers.22.self_attn.q_proj.weight
- Weights changed (bf16): 191645

Removes the content-free 'option letter token' vote from the answer-copy head L22.H12 only. Change: model.layers.22.self_attn.q_proj.weight rows 1536:1664 (head 12): W_h <- W_h - 1.5 w w^T W_h (rank 1), w = normalize(q_norm.weight * u), u = mean over 240 DEV questions of (mean group-3 k_norm key at the option letter tokens '(X' minus mean key at the option end and newline tokens). Mechanism: L22 is NoPE; H12 (the head whose attention picks the answer, +2.9 logits to gold) puts 12% of its attention on letter tokens whose keys precede the option text (causal), i.e. a content-free slot vote (B over-picked). Lens: dropping letter tokens from H12's option vote +5/-2 (recorded attention) / +8/-0 (recomputed q.k), letter-token mass .116 -> .026 at this strength. Evidence (paired): DEV3 256 q: acc +1.17 (SE 0.87), 4/1, dlogp +0.022 (SE 0.023), gold rank -0.055 (SE 0.037). HOLDOUT 512 q (measured once): acc 0.00 (SE ~0.73), 7/7, dlogp +0.020 (SE ~0.017), gold rank -0.025 (SE ~0.027). Accuracy-neutral.

## Selection (hidden questions) and decision

Parent accuracy on the 800 selection questions: **36.38%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| claude/l12h13_half | 36.12 | -0.25 | 0.75 | -0.33 | 17 | 19 | +0.2496 | 262144 |
| ed_b/letterproj_L22_h12 | 35.88 | -0.50 | 0.5 | -1.00 | 6 | 10 | +0.0133 | 191645 |
| ed_a/ed_a_g3_prior17 | 35.75 | -0.62 | 0.78 | -0.79 | 17 | 22 | +0.1094 | 34233 |
| ed_b/letterproj_L22_all | 35.38 | -1.00 | 0.59 | -1.69 | 7 | 15 | +0.0268 | 2964771 |
| claude/l12h13_ablate_retest | 35.12 | -1.25 | 1.22 | -1.02 | 43 | 53 | +0.5136 | 262144 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

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

### ed_a

# ed_a, generation 3: report

_Saved by the orchestrator: ed_a's tool environment refused to write this file, so the text below is ed_a's final reply._

## Summary
- **The suggested direction does not work in-context.** Inside the evaluation prompt, an option text's likelihood says nothing about the gold answer. The cloze score (the option text as the answer to the bare question) does carry some information.
- **Late attention adds little.** It holds at most about 1.5 points of accuracy beyond the final answer.
- **New mechanism: an off-by-one in L22.H12's letter output.** It is real but worth at most about 0.5 points.
- **Candidate `ed_a_g3_prior17`.** It removes the letter prior written by 17 near-constant late MLP neurons.
  - On HOLDOUT (1,000 questions, measured once): **+0.80 points (SE 0.65, z 1.23)**, not significant.
  - Log-prob of the correct letter: **+0.133 (SE 0.013)**.

## What the model gets wrong
I analysed the 946 DEV questions from pass 1. The parent scores 33.5% on them.

**Late attention holds almost no extra information about the gold option.**
- L22.H12 puts 42% of its answer-position attention on option text tokens and 45% on the separate "\n" token after each option.
- Its attention argmax picks gold 34.5% of the time, against 33.5% for the final letter. When the argmax is right, the final answer is right 86% of the time.
- A cross-validated logistic model over the attention of all 112 heads in layers 10–22 reaches 35.1%. A constant letter debias reaches 34.2%.
- Everything after selection is therefore worth at most about 1.5 points. The errors are knowledge errors.

**Other hypotheses I checked, all null:**
- The correctness signal does not get weaker for late option slots or beyond the 256-token sliding window.
- A mean-difference probe on the residual stream does better at h14 than at h21, and neither beats H12.
- Negated questions are only 5% of the set and are not special.

**Unused option-text knowledge (the suggested starting point).**
- *In-context* (pass 2, 400 questions): the log-probability of each option's text inside the rendered prompt picks gold 9–10% of the time (chance is about 11%). On questions the letter answer gets wrong it is at chance (12%). The list format and the earlier options dominate those predictions.
- The logit lens at the end of the question is also at chance.
- So the network does not represent this knowledge at the option tokens, and there is no existing signal to route into the belief.
- *Out-of-context* cloze (pass 10, 107 DEV questions, "The answer is <option text>" after the bare question): the mean log-probability per token picks gold 19.6% of the time.
  - On letter-wrong questions, excluding the letter's pick, it finds gold 18.8% of the time (chance 12%).
  - Adding 0.5 × its z-score to the letter z-scores gave 35.5% → 40.2% (7 wrong→right, 2 right→wrong). That weight was the best of 6 tried in-sample, so this is suggestive only.
  - The knowledge exists, but the model only computes it when it generates the answer itself. Using it would need new circuitry, not a reweighting.

**New mechanism: the off-by-one in L22.H12's letter transfer (code/013–016, 024).**
- Value vectors at the newline after option s are read out through H12's OV circuit and the letter rows.
- The diagonal of this transfer is strong but uneven. Per unit of attention, A is 16, B 10.5, G 31.5 and H 30.
- There is also a systematic write to letter **s+1**, at 0.2 to 0.6 times the diagonal.
- Cause: the newline token predicts the next token "(s+1". In the logit lens at h21, "(s+1" shows on the super-diagonal, and the "(X" embedding rows have cosine 0.65 with the bare letter rows.
- **Fixes that failed:**
  - Projecting the "(X" subspace out of the value removes the diagonal too, because the option's own letter identity lives in the same subspace.
  - Removing the excess with oracle knowledge, in a screen that reads the letter logits directly off recorded activations, is worth only +0.5 points.
  - A letter-space correction of H12's output costs 1.8 points in that screen.
- I did not submit an edit for it.

## What I changed and why
**Edit.**
- Tensors: `model.layers.22.feed_forward.w2.weight`, columns 190, 1449, 2820, 4543, 5653, 2212, 3998.
- `model.layers.23.feed_forward.w2.weight`, columns 1047, 4597, 1903, 6518, 4531, 514, 3073, 2189, 1439, 7466.
- For each column, I projected out the 9-dimensional centred letter-readout subspace (span of `embedding_norm.weight` × E[A..J], minus their mean). This is done in post_ffn_norm-scaled space at strength 1; the rest of each column is unchanged.
- 34,233 weights differ in bf16.

**Why these neurons.** They fire almost constantly at the answer position and write a fixed preference: A, B, C and F up, H, I and J down. This set is the layer 22/23 subset of generation-1 ed_a's list. Removing their letter write takes out a content-free prior and leaves the routing heads and the suppressor banks untouched.

**Alternative rejected on DEV.** Edit C1 zeroed the letter readout of the L22 C–J leader suppressors (generation 1). On DEV2 it gave −0.78 (SE 1.24) and log-prob −0.28.

## Evidence
Order in which I saw results:
1. Direct-logit screens on DEV-946 (letter logits read directly off recorded activations).
2. C2 (the submitted edit) and C1 on DEV2[0:256].
3. C2 on DEV pass1[0:256].
4. I chose C2.
5. C2 on HOLDOUT[0:1000], in four passes, each measured once. I chose nothing after seeing them.

All gains are paired (edited minus parent). "a/b" counts questions that went wrong→right / right→wrong.

| Edit | Questions | Accuracy gain (points) | a/b | Log-prob of correct letter |
|---|---|---|---|---|
| C2 | DEV2[0:256] | −0.39 (SE 1.30) | 5/6 | +0.128 (SE 0.025) |
| C2 | DEV pass1[0:256] | +0.39 (SE 1.30) | 6/5 | +0.136 (SE 0.027) |
| C1 | DEV2[0:256] | −0.78 (SE 1.24) | 4/6 | −0.276 (SE 0.039) |
| **C2** | **HOLDOUT 1,000** | **+0.80 (SE 0.65, z 1.23)** | **25/17** | **+0.133 (SE 0.013)** |

**What the HOLDOUT shows.**
- Accuracy goes from 35.8% to 36.6%. The gold option's rank improves by about 0.08.
- By gold letter, accuracy rises on H (.21→.27), I (.20→.24) and J (.19→.23) and falls on B (.43→.36).

**The screen is faithful, but sampling noise dominates.**
- The direct-logit screen predicted the log-prob gain (+0.14) and 24 of the 28 real prediction flips on pass1[0:256], with 96% going to the same letter.
- But its accuracy prediction swings between subsets: +0.0 on pass1[0:256] and +2.6 on pass1[256:946].

**Conclusion.** This is the same picture as generations 1 and 2. Letter-prior edits robustly improve the correct letter's log-probability and buy about +0.5 to 1 point of accuracy, which is below the selection bar.

## What I would try next
1. **Stop editing the letter stage.** Three generations and six or more variants agree on a ceiling of about 1 to 1.5 points. Here, the late-attention ensemble ceiling is +1.6, a constant debias +0.7 and the off-by-one oracle +0.5.
2. **Use the out-of-context cloze knowledge.** It is the only extra knowledge source found so far: about 19% gold on letter-wrong questions, and a combination gain that looks large on 107 questions.
   - First, measure it on 300 or more questions with a cross-validated weight.
   - If it holds, find where the model computes "the answer to this question" when it generates the answer itself. That is the residual at assistant-turn positions in the cloze texts; record it with `positions: all`.
   - Then test whether any position in the multiple-choice prompt carries the same representation, for example the question's last token at middle layers.
3. **Possible fix for the off-by-one.** It could be repaired in the value circuit if the next-token-prediction component of the newline residual can be separated from the option's own letter identity, for example by the layer at which each appears.

Files are in /home/user/weight-space-experiments/runs/gen_003/editors/ed_a:
- notes.md
- code/001–027 (scripts)
- work/spec_C2.json (the submitted edit's spec)
- candidates/ed_a_g3_prior17

### ed_b

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

