TASK. You are given the weights of a language model, LFM3-1.7B (Liquid AI, October 2026). Text part: 24 layers of width 2,048. Each layer has one token mixer and one MLP:
- 13 gated short-convolution layers (0, 1, 3, 5, ..., 23): `conv.in_proj` (2048 -> 6144, split into three 2048-wide parts B, C, x), a depthwise causal convolution `conv.conv` (kernel 7) and `conv.out_proj` (2048 -> 2048);
- 6 full-attention layers (2, 6, 10, 14, 18, 22) and 5 sliding-window attention layers (4, 8, 12, 16, 20; window 256): 16 query heads and 4 key/value heads of size 128, RMS-normed queries and keys (`q_norm`, `k_norm`), a per-head sigmoid output gate (`self_attn.gate_proj`, 2048 -> 16) and `o_proj` (2048 -> 2048, input = 16 heads x 128);
- a SwiGLU MLP in every layer: `feed_forward.w1` (gate) and `feed_forward.w3` (up), 2048 -> 7680; `feed_forward.w2` (down), 7680 -> 2048;
- RMSNorm in sandwich form: `operator_norm` before the mixer, `post_operator_norm` after it, `ffn_norm` before the MLP, `post_ffn_norm` after it (both post-norms are applied before the residual add).
Vocabulary 128,000; input embedding and output layer tied; the final norm is called `model.embedding_norm`.
Your job is to make it better at answering broad multiple-choice questions on knowledge and reasoning (MMLU-Pro: 10 options, 14 subjects from law and history to physics and math) by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it answers wrongly, and repair that with targeted changes.

THIS IS ONE GENERATION OF A GENETIC LOOP. You are editor `claude` in generation 2. Several editors work in parallel on the same parent model; each submits up to 2 candidate edits. The candidates are then compared with the parent on hidden questions, and the best one becomes the parent of the next generation, but only if its gain is clearly significant. Otherwise the parent survives. The parent of this generation is `base`; "original" below always means the parent. It is the unedited model.

HOW IT WILL BE MEASURED. Every candidate is evaluated with this fixed protocol. The user message is the question, a blank line, the options as lines "(A) ...", "(B) ...", ..., a blank line and "Answer with the letter of the correct option." It goes through the chat template with thinking disabled (enable_thinking=False), and the assistant turn is opened with "The answer is (". The model's answer is the option letter with the highest logit at the next token (only the letters of the question's options compete). Weights are cast to bfloat16. The score is accuracy. Selection compares the candidate with the parent on the same hidden MMLU-Pro questions, as a paired difference with its standard error; the bar is z >= 2. Later, champions are also evaluated on untouched held-out MMLU-Pro questions and on GPQA Diamond (4 options, graduate-level science), so edits that exploit something specific to 10-option MMLU-Pro will not carry over. You never see the selection or held-out questions. The config, tokenizer, chat template and protocol are fixed: only weight values can change (same tensor names and shapes).

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it answers wrongly, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Repairing a specific failure mechanism is in scope: for example, if the model systematically over-picks early letters regardless of content, or a component suppresses the correct answer once it has been retrieved, finding and fixing what drives that counts. Out of scope, and refused by save_edit: the final norm (`model.embedding_norm.weight`), the input embedding / output layer (`model.embed_tokens.weight`), and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for.

WHAT YOU HAVE.
1. Python over the weights (no GPU, no internet). Your folder is `/home/user/weight-space-experiments/runs/gen_002/editors/claude`: `edited/` (your current edit), `work/` (scratch, forward-pass outputs), `code/` (your scripts), `candidates/`, `notes.md`. Run Python with `/home/user/weight-space-experiments/runs/gen_002/editors/claude/py <script>` (working directory = your folder). Write every script you run to `code/NNN_short_name.py` (numbered in order) so the code is kept, and state its hypothesis in its first comment line. Use `-c` only for one-line checks. There is no persistent kernel: save intermediate arrays in `work/` as .npy. In a script, `from wsp.editor import *` gives you:
   - NAMES and SHAPES (all tensors);
   - load(name), a float32 torch tensor: your current edit if there is one, else the original (= parent);
   - load_original(name) and load_base(name) (the unedited LFM3-1.7B);
   - save_edit(name, tensor) (it reports how many entries change after bfloat16 rounding), drop_edit(name), edited();
   - submit_candidate(label, description);
   - tok (the tokenizer);
   - questions() (the practice questions);
   - render(qid) (the exact evaluation string of a practice question);
   - LETTER_IDS (token ids of A..J as the token after "The answer is (").
   Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, safetensors, tokenizers. Gradients are disabled. The machine has 4 CPU cores and 15 GB of RAM, shared with the forward-pass server (about 4 GB) and the other editors. Keep each script under about 3 GB and a few minutes of compute: load the tensors you need, not the whole model.
2. Practice questions with answers (none of them is used for selection or held-out evaluation): questions() returns 3,007 MMLU-Pro questions, {id: {id, category, src, question, options, answer, answer_index}}, across all 14 categories.
3. A forward-pass tool with a budget of 10 forward passes in total. Write a request as JSON to a file and run `/home/user/weight-space-experiments/runs/gen_002/editors/claude/fwd <file.json>`. A request that fails or is refused does not count. Each request counts as one pass, whatever its size:
 (a) SCORE: {"type": "score", "model": "original" | "edited" | "both", "questions": ["<practice id>", ...] or "sample": {"n": 256, "seed": 0, "categories": [...] (optional)}, "activations": {"hidden_states": [0..24], "mlp": [0..23], "inputs": [module names], "outputs": [module names]} (optional)}
    - This runs the evaluation protocol on at most 512 practice questions per model (at most 256 with "both").
    - You get accuracy (overall and by category), mean log-probability of the correct letter, margin, how much probability the full vocabulary puts on the option letters, and predicted versus true letter counts.
    - "both" runs the parent and your current edit on the same questions and adds the paired difference (edited - original) with its standard error, plus how many questions went wrong->right and right->wrong.
    - Per-question rows (prediction, correctness, the probabilities of all letters) are in `work/fwd_<k>/score_<k>_<model>.jsonl`.
    - With "activations", these are saved at the answer position (the last token) for every question, with its grade, in `work/fwd_<k>/score_<k>_acts/<model>/`: the residual stream (0 = embeddings, l = after layer l), the MLP's w2 inputs (7,680 per layer), or the input/output of any named module (e.g. "model.layers.14.self_attn.o_proj" gives the 16 heads x 128 before mixing).
 (b) READ (teacher-forced): {"type": "read", "model": "original" | "edited", "texts": ["raw text", ...] or "questions": [ids] (rendered evaluation strings) with optional "suffixes": [...], "return": {"token_logprobs": true, "topk": k (<= 50), "hidden_states": [...], "mlp": [...], "inputs": [...], "outputs": [...], "positions": "last" | "mean" | "all", "attn_layers": [attention layer indices], "attn_last": [attention layer indices]}}
    - This reads up to 1,024 texts, at most 200,000 tokens in total and 4,096 per text.
    - Results are .npy files with meta.json in `work/fwd_<k>/`.
    - With "all" positions, activations are concatenated over texts, with offsets.
    - "attn_layers" returns full attention matrices for the first 4 texts only.
    - "attn_last" returns, for EVERY text, the attention probabilities from its last token (the answer position) to all positions, for each head. Attention layers are 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 and 22.
 Timing: about 0.75 s per question per model. Results for the parent are cached, so a "both" pass without activations only costs the edited model's time. One full 256-question "both" pass takes a few minutes, plus waiting while other editors' passes run (one queue for everyone). `fwd` waits up to 9 minutes; if the pass is not done, it prints `fwd wait <id>`, which you run to keep waiting. `/home/user/weight-space-experiments/runs/gen_002/editors/claude/fwd status` shows the queue and your budget.

HOW TO WORK.
1. Diagnose first. Look at how the model fails on practice questions:
   - the correct answer ranked second (near misses) versus confidently wrong answers;
   - systematic letter preferences (compare predicted and true letter counts);
   - categories where it is weak;
   - questions where a middle layer already "knows" the answer and later layers lose it. You can read the residual stream through the output layer, since the embedding is tied; apply the final norm first.
   Go after the largest failure category you can explain.
2. Localize causally. That a component is more active on correct answers is not enough (correct answers come from easier questions). Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. Accuracy over n questions has a standard error of about 50/sqrt(n) percentage points (about 3.1 points at 256). Compare original and edited on the SAME questions: a "both" pass reports the paired difference with its standard error, which is much smaller than that of two separate measurements. Keep an edit only if its gain is clearly larger than its standard error on questions that were not used to choose it: split the practice questions into a DEV part you analyse and a HOLDOUT part you only use to confirm. Choose your final candidate(s) using DEV only, then measure each on HOLDOUT once; do not pick among variants by their HOLDOUT results. In generation 1, editors' gains measured after choosing among variants shrank to about zero on the hidden selection questions. The continuous mean log-probability of the correct letter is a much less noisy signal than accuracy, so use it to screen. Edits that change a tensor by a fraction of a percent usually have effects too small to see: test strengths large enough to matter, and back off if they hurt.
4. Plan your forward passes. For example: one diagnostic score pass with activations, a few read passes or "both" passes for dose-response, and a final "both" pass on HOLDOUT questions you have not used, to confirm.
5. Work efficiently.
   - Print compact summaries, never whole arrays.
   - Never print raw lists of vocabulary tokens (the vocabulary contains offensive strings); print token ids, letters or common words instead.
   - Keep `notes.md` up to date: findings with numbers, edits tried and their measured effect, the current best edit. Before every pass, write the exact current edit (tensors, what was changed, strength) and the question set.
   - Copy good edits (for example `cp -r edited work/best_1`).

RULES. Not allowed:
- (a) Gradient descent or any other training: no backward passes, autograd or optimizers. No fitting of weights to a loss on data, whether iterative or in closed form (for example, solving a weight matrix by least squares so that it reproduces target outputs).
- (b) Running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in Python. Reading the residual stream through the tied output layer is the one exception: it is one matrix product.
- (c) Internet access or search.
- (d) Looking outside your folder and the API: no reading of other editors' folders, the data files or the GA's state.
Linear read-outs of RECORDED activations (logit lens, direct logit attribution, and arithmetic on them) count as analysis and may be used to find components and to pre-screen ideas, but every effect you claim must be measured with the forward tool. Allowed: any analysis of the weights and of forward-pass results, any use of the practice questions, and any direct edit of the weights that you compute and that is in scope. That includes directions computed from recorded activations as means, differences or principal components, projections, rank-one edits, and switching or rescaling specific neurons, heads or channels.

HOW YOUR WORK ENDS. Use your forward passes. Then submit your best edit or edits with submit_candidate(label, description), at most 2. That is the candidate with the best measured paired gain, on HOLDOUT if you measured it there, even if the gain is not statistically significant; do not submit an empty edit. In each description, say what was changed (tensors, indices, strength), the mechanism, and the measured paired gain with its standard error and the questions it was measured on. Finally, write a short report to `/home/user/weight-space-experiments/runs/gen_002/editors/claude/report.md` and reply with it: what the model gets wrong, what you changed and why, the evidence, and what you would try next with more passes. The report is read by the next generation's editors.

HISTORY OF THE LOOP.
No edit has been accepted yet: the parent is the unedited LFM3-1.7B.

Generation 1 (parent base, parent accuracy 35.0 on its hidden selection questions). Winner: none, the parent survived. Candidates:
- claude/equalize_boost: +0.50 points (SE 0.9, z +0.56), 24398 weights. Edit: scale the DIRECT LETTER READOUT (the component of a w2 column that the tied letter rows read through post_ffn_norm and the final norm; a 9-dim subspace) by 0.5 for (a) the 8 letter-specific "leader suppressor" neurons of layer 22's MLP, model.layers.22.feed_forward.w2.weight columns C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310, and (b) 4 early-letter "booster" neurons: L22 column 2212 (A), L23 columns 1685 (A), 2595 (A), 1439 (B). The rest of each column is unchanged. 24,398 weights differ in bf16.
Mechanism: after head L22.H12 moves the chosen option to the answer position, the L22 MLP suppresses the currently leading letter through one neuron per letter (-3 to -4.6 lo
- claude/equalize_l22: +0.50 points (SE 0.73, z +0.68), 16254 weights. Edit: the same as equalize_boost but only part (a): the direct letter readout of the 8 layer-22 late-letter leader suppressors (model.layers.22.feed_forward.w2.weight columns 1812, 464, 7357, 6492, 3344, 397, 2786, 3310) is scaled by 0.5. 16,254 weights differ in bf16.
Evidence: screen256 DEV +1.56 (SE 0.95), +5/-1; HOLDOUT 512: +0.98 (SE 0.85), +12/-7, dlogp(correct) -0.153.
- ed_a/ed_a_v1_letterbias12: -0.50 points (SE 0.96, z -0.52), 24269 weights. Removes the answer-letter position prior written by 12 near-constant MLP neurons. Change: for w2 columns (layer,neuron) (20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820), project out (in post_ffn_norm-scaled space) the 9-dim subspace spanned by the centered letter readout directions wn*E[A..J] (column -> (I-QQ^T)(pw*col)/pw); strength 1 (full removal). Mechanism: these neurons are ~constant at the answer position (cv<0.6) and write a fixed preference A,B,C,F >> H,I,J (orig bias in centered logits [1.15 1.66 .57 .16 .03 1.27 -.05 -1.14 -2.09 -1.58]); removal flattens it (S: I -2.59 -> -0.34) with no measurable self-repair (di
- ed_a/ed_a_v2_letterbias27h: -0.50 points (SE 1.12, z -0.45), 306498 weights. Removes the answer-letter position prior written by near-constant components at the answer position. Change: centered letter-readout subspace (9-dim span of wn*E[A..J] minus mean) projected out (in post-norm-scaled space, strength 1) of: 27 MLP w2 columns (layer,neuron) (20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820),(22,4543),(21,218),(23,6518),(23,4531),(22,5653),(23,514),(23,3073),(23,2189),(21,4481),(22,2212),(22,3998),(23,1439),(23,7466),(21,3455),(21,4902), and head L22h6 (o_proj input cols 768:896). Mechanism: parent over-picks A/B/C/F and starves H/I/J (pred A85 B99 .. I20 J14 vs true A69 B48 .. I41 J49 on 512 DEV)
- ed_b/ed_b_bias16_a10: -1.33 points (SE 0.94, z -1.41), 32156 weights. alpha = 1.0. Mechanism: the parent over-picks early letters (DEV-512 preds A..J 94 95 78 58 40 54 38 28 15 12 vs gold 59 49 51 68 47 50 42 69 39 38; acc by gold letter A .58 -> I .21). Per-neuron direct logit attribution onto the option-letter unembeddings (through final norm) shows a set of late MLP neurons with near-constant activation whose letter push is an ordinal 'early up / late down' profile (largest: L20 n1000, L22 n190, L17 n2401). Selected by: mean push along the measured offset > 0.08 logit, sd < mean, |help on correct letter| < 0.03 (DEV[:512]). Edit: for each of the 16 neurons, w2[:, j] -= alpha * P w2[:, j], P = projector onto the centered letter-readout subspace (letter unemb
- ed_b/ed_b_bias16_a15: -1.50 points (SE 1.21, z -1.24), 32337 weights. alpha = 1.5. Mechanism: the parent over-picks early letters (DEV-512 preds A..J 94 95 78 58 40 54 38 28 15 12 vs gold 59 49 51 68 47 50 42 69 39 38; acc by gold letter A .58 -> I .21). Per-neuron direct logit attribution onto the option-letter unembeddings (through final norm) shows a set of late MLP neurons with near-constant activation whose letter push is an ordinal 'early up / late down' profile (largest: L20 n1000, L22 n190, L17 n2401). Selected by: mean push along the measured offset > 0.08 logit, sd < mean, |help on correct letter| < 0.03 (DEV[:512]). Edit: for each of the 16 neurons, w2[:, j] -= alpha * P w2[:, j], P = projector onto the centered letter-readout subspace (letter unemb

Report of editor claude (generation 1):
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


Report of editor ed_a (generation 1):
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
- **Dose-response is smooth.** In the direct estimate, strength 0.25 / 0.5 / 0.75 / 1.0 / 1.25 gives 35

Report of editor ed_b (generation 1):
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

#


SUGGESTED STARTING POINT (to diversify the generation; abandon it if the evidence points elsewhere): Near misses: questions where the correct letter is ranked second. Decompose the correct-minus-chosen margin by component and find what systematically flips the top two.