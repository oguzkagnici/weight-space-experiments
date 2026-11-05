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
