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
