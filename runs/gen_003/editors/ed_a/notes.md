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
