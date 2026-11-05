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
