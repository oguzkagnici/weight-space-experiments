# Notes of editor ed_a, generation 3

## Setup
- Split (code/001, seed 3): work/split.json, DEV = 2007 practice ids, HOLDOUT = 1000 ids (never analysed; only final confirmation).
- Rendered prompt: options are lines "(X) text\n"; "(A".."(J" are single tokens; option text end token often fused with "\n".

## Plan / hypothesis 1
Gen-2 ed_b: L22.H12 attention argmax picks gold 40.6% (DEV-160) vs final accuracy ~35%. If the attention selection is
more accurate than the emitted letter, some of the loss is in the letter-transmission (value/OV) path, a mechanical
failure that could be repaired. Pass 1 tests this at scale.

## Pass log
- Pass 1 (read, original, DEV first 946 questions = work/pass1_ids.json, 190k tokens): attn_last all attention layers,
  hidden 12,16,20-24 last, mlp 22,23, o_proj inputs L14/L20/L22, topk 50. No edit staged.

## Pass 1 results (code/004-006, original, DEV 946)
- acc 0.335 (topk letters); preds A..J 151 162 157 107 64 105 85 45 39 31 vs gold 111 108 87 114 96 92 92 85 90 71.
- Category acc: biology .62, economics .56, engineering .46, psychology .42, other .39; law .18, physics .23, business .24, math .25.
- L22.H12 puts 42% of its answer-position attention on option text tokens and 45% on the separate "\n" token after each option.
  Its argmax option (span+newline) is gold 34.5% (vs final 33.5%); agrees with final 75.5%; when H12 argmax is right final is right 86%.
  => hypothesis 1 (selection more accurate than emitted letter) is FALSE: no transmission loss worth fixing.
- H12 attention argmax counts are themselves position-biased: A..J 116 165 155 108 58 73 100 39 69 63.
- CV option-level logistic on log attention masses of all 112 heads of L10-22: 35.1% (final 33.5%); + final logits 35.1%;
  letter-dummy debias of final logits: 34.2%. => late attention holds almost no extra info on gold. Content/knowledge bound.
- Lens acc: h16 .20, h20 .24, h21 .24, h22 .26, h23 .33, h24 .34.
- Confidence: pmax>.7 (n=413) acc .51; pmax<.5 (n=328) acc .17.

## Hypothesis 2 (suggested start): unused option-text knowledge
Does the in-context log-prob of each option's text (inside the rendered prompt) pick gold on questions the letter
protocol gets wrong? If yes, is there a residual feature at option tokens encoding it that H12's keys underuse?
- Pass 2 (read, original, DEV first 400 of pass1 ids = work/pass2_ids.json): token_logprobs, hidden 14 & 21 all positions,
  L22 k_norm and v_proj outputs all positions. No edit staged.

## Pass 2 results and analyses (code/007-021)
- In-context option likelihood (mean/sum/first-token log-prob of each option's text inside the rendered prompt) picks gold
  9-10% (chance ~11%); on final-wrong questions 11-12% vs chance 12%. Combining with final logits never helps. => the
  "unused option-text knowledge" is NOT present in-context (previous options/list format dominate the prediction).
- Logit lens at the question end (h14/h21) scoring option tokens: ~chance (13-17% on wrong q vs 12%).
- Mean-difference belief probe on raw residual at option-end newlines: worse than H12 (h14 .23, h21 .14).
- Sliding window: no AUC loss for gold options > 256 tokens after question start.
- Negated questions: only 5%, not special.
- NEW MECHANISM (code/013-016): L22.H12 OV "letter transfer" from the newline after option s: strong diagonal but
  (i) non-uniform gain by slot (per unit attention, A 16 B 10.5 C 19 D 18 E 25 F 24 G 31.5 H 30 I 21 J 13) and
  (ii) an OFF-BY-ONE write: the newline after option s also writes letter s+1 at ~0.3-0.5x the diagonal
  (B->C, D->E, E->F, F->G, H->I, I->J). Cause: the newline residual predicts the next token "(s+1" (logit lens at h21 shows
  "(s+1" on the super-diagonal) and E["(X"] has cos .65 with the bare letter rows. Projecting the "(X" subspace out of
  the value kills the diagonal too (self identity lives in the same subspace). Oracle removal of the off-by-one excess
  (all L22 heads) in a DLA screen: +0.5 pt only (3 w->r / 1 r->w of 400). Equalising H12's diagonal alone: -1 pt.
  => real but cheap: not worth an edit.
- Ceiling summary: H12 attention argmax .345 vs final .335; when argmax right final right 86%. All letter-stage fixes
  are bounded by ~+1.5 pt on this DEV.
- DLA screens on DEV-946 (direct effect only, fixed attention/norms) of earlier mechanisms (code/019-021):
  C-J leader suppressors (L22 n1812,464,7357,6492,3344,397,2786,3310) readout f=0.5: +0.6; f=0: +2.0 (53/34), logp -1.1;
  L22/L23 near-constant prior writers (gen1 ed_a list, 17 cols) removed: +1.9 (39/21), logp +0.14;
  both: +1.6 (not additive: both counter the early-letter prior).

## Edits for the real network (code/022_make_edit.py: remove fraction f of the centered 9-dim letter readout of w2
## columns in post_ffn_norm-scaled space)
- C2 = work/spec_C2.json: L22 cols 190,1449,2820,4543,5653,2212,3998 and L23 cols 1047,4597,1903,6518,4531,514,3073,
  2189,1439,7466, f=1. (tensors model.layers.22/23.feed_forward.w2.weight)
- C1 = work/spec_C1.json: L22 C-J suppressors, f=1.
- DEV2 = DEV minus pass-1 questions (work/dev2_ids.json, 1061 q). HOLDOUT untouched.
- Pass 3: both, DEV2[0:256], edit C2.
- Pass 4: both, DEV2[0:256] (same q as pass 3), edit C1 (queued while pass 3 waited; edits are snapshotted).

## Passes 3-4 results (real network, DEV2[0:256], paired edited - original)
- C2 (prior writers L22/L23, 17 cols, f=1): acc -0.39 (SE 1.30), 5 w->r / 6 r->w, logp(correct) +0.128 (SE 0.025),
  gold rank -0.063 (SE 0.043), entropy .964 -> .992; preds A..J 40 31 47 32 23 17 21 16 17 12 (orig 38 45 52 26 22 20 22 11 12 8).
  The logp gain matches the DLA screen (+0.14) => no self-repair; accuracy does not follow.
- C1 (C-J suppressors, f=1): acc -0.78 (SE 1.24), 4/6, logp -0.276 (SE 0.039), rank -0.039 (SE 0.039), entropy .83.
- Pass 5: both, pass1_ids[0:256] (DEV questions used in the DLA screen), edit C2: checks whether the screen's accuracy
  prediction is faithful there (if yes, the DEV2 shortfall is sampling noise).
- Pass 5 result (C2, pass1_ids[0:256]): acc +0.39 (SE 1.30), 6/5, logp +0.136 (SE 0.027), rank -0.051 (SE 0.044).
  code/023: the DLA screen predicted 0.0 (7/7) on exactly these questions; 24 of its 42 predicted flips also flip in the
  real network (96% to the same letter) => the screen is faithful; the +1.9 of the 946-question screen comes from
  pass1_ids[256:946] (+2.6 screen). Pooled DEV evidence for C2 ~ +1 point, logp +0.13 robust.
- code/024: a letter-space off-by-one correction on H12's output (kappa_s = measured super/diagonal ratio,
  .2 .56 .29 .56 .39 .59 .3 .31 .48) HURTS in the screen (c=1: -1.8 pt) -> abandoned; off-by-one stays a finding only.

## Decision (DEV only): candidate = C2. C1 rejected (DEV2 -0.78, logp -0.28).
- Holdout plan: C2 on HOLDOUT[0:1000] in 4 passes (6-9), each measured once; no further choice among variants.
- Current edit staged = C2 (work/spec_C2.json, copy in work/edit_C2).
- Pass 10 (queued after holdout passes): read, original, cloze texts (bare question as user turn, assistant 'The answer is <option text>') for 107 DEV questions from pass1_ids[256:] (work/cloze_meta.json), token_logprobs. Diagnostic for next generation only; does not affect candidate choice.
- Pass 6 (HOLDOUT[0:256], C2): acc +1.17 (SE 1.30), 7/4, logp +0.127 (SE 0.025), rank -0.031 (SE 0.044).
- Pass 7 (HOLDOUT[256:512], C2): acc +1.17 (SE 1.41), 8/5, logp +0.144 (SE 0.027), rank -0.102 (SE 0.046).
- Pass 8 (HOLDOUT[512:768], C2): acc +0.00 (SE 1.36), 6/6, logp +0.105 (SE 0.023), rank -0.102 (SE 0.041).
- Pass 9 (HOLDOUT[768:1000], 232 q, C2): acc +0.86 (SE 1.06), 4/2, logp +0.160 (SE 0.026), rank -0.082 (SE 0.048).
- POOLED HOLDOUT 1000 (code/027): parent 35.80 -> 36.60, diff +0.80 (SE 0.65, z 1.23), 25 w->r / 17 r->w;
  logp(correct) +0.133 (SE 0.0125). Gold-letter accuracy: B .43->.36, H .21->.27, I .20->.24, J .19->.23, others ~flat.
  Preds A..J 174 164 162 119 96 99 73 50 34 29 -> 177 122 152 128 101 93 69 60 56 42 (gold 134 104 115 115 74 95 83 101 95 84).
- Pass 10 (cloze, 107 DEV q, code/026): out-of-context cloze mean-logprob picks gold 19.6%; on letter-wrong q (excluding
  letter pick) 18.8% vs chance 12%; letter_z + 0.5*cloze_sum_z: 35.5 -> 40.2% (7/2) in-sample over 6 weights (suggestive).
  Contrast: in-context option likelihood (pass 2) is at chance -> the knowledge exists only in a generative frame.

## Final
- Submitted candidate `ed_a_g3_prior17` = C2 (34,233 bf16 weights; L22 w2 cols 190,1449,2820,4543,5653,2212,3998;
  L23 w2 cols 1047,4597,1903,6518,4531,514,3073,2189,1439,7466; centered letter-readout projection, strength 1).
  HOLDOUT 1000: +0.80 (SE 0.65), logp +0.133 (SE 0.013). Only one candidate (C1 was worse on DEV).
- report.md written. All 10 passes used.
