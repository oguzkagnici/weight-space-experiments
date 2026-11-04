# Notes of editor ed_b, generation 2


## Setup
- Split (code/002): seeded perm (seed 2002) of 3007 practice ids; HOLDOUT = first 768, DEV = rest (2239). In work/split.json.
- Direction (suggested start): is there a correct-vs-incorrect signal at option positions upstream of L22, and does L22.H12 read it via keys?

## Pass 1 (read, parent, DEV[:160]) -- current edit: none
- hidden_states 14,18,20,22,24 all positions; L22 k_norm outputs; attn_last all attention layers.
- Results pass 1 (code/004-009): parent acc 43.1% on these 160. L22.H12 attends mostly to option END tokens (0.65 of its mass; letter tokens 0.12).
  H12 attention argmax picks gold 40.6%; agrees with the model's final pick 72.5%. No positional prior of note in H12 attention (argmax counts ~ gold counts);
  the A/B/C-up, I/J-down letter bias arises AFTER selection (final centered logits A .54 B 1.13 C 1.07 ... I -1.51 J -1.63).
- Correctness probes at option ends (5-fold CV by question, within-question centered): diff-of-means h14 40.0%, h18 32.5%, h20 30.6%, h22 25.6%; LR(C=1e-3) h14 37.5, h20 41.2, h22 35.0.
  On MODEL-WRONG questions the probes pick gold 8-12% (chance ~10%); span-mean probes 13-16%. => no linear unread correctness signal at option positions.
  Ensembles final logits + b*probe or + b*log attn(H12) give <= +2/160 questions. NEGATIVE for "strengthen the key read": H12 already reads what the options carry.
- Next: localize WHERE (layers 4..16) the option-end belief is written and whether a specific head's contribution is mostly signal (amplify -> SNR).

## Pass 2 (read, parent, 180 DEV[160:700] questions <=320 tokens, list in work/p2_qids.json) -- current edit: none
- hidden_states 4,8,10,12,14,16 all positions; o_proj inputs L8,10,12,14; attn_last L22.
- Results pass 2 (code/012-015): option-end belief forms between h8 and h14 (sep along pass-1 d14 direction: h4 .02, h8 .25, h10 .81, h12 .89, h14 1.00, h16 .99; CV-DM acc h12 33%, h14 31%; H12 acc on this set 32%).
  Heads writing it: L12.H13 (diff .50, SNR .85, alone 29%), L10.H4, L8.H1, L12.H8 -- but they correlate 0.5-0.83 with the rest of the stream; linear amplification screen (x0..x5) changes belief acc by <= +1/180. The bulk (diff ~4) is written by MLP/conv; signal is one coherent, redundant belief.
  L14.H12 is already a selection head at option ends (argmax acc 41.9% on pass-1 set, mass .53); every selecting head is at chance on model-wrong questions.
  CONCLUSION: the content path is saturated; errors are knowledge (gold at chance among wrong answers). Abandon the key-path direction.
- Pivot: the post-selection letter stage (H12 attention argmax counts are flat vs gold, final picks are A/B/C-heavy).

## Pass 3 (score, parent, DEV[700:1212], 512 q) -- current edit: none
- activations: hidden 18,20-24; mlp 20-23; o_proj input L22.
