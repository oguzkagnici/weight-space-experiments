# Notes of editor ed_b, generation 1


## Setup
- code/001_split.py: practice split stratified by category, seed 1234 -> work/split.json: DEV 1499, HOLDOUT 1508.

## Pass 1 (diagnostic)
- Edit: none. model=original, questions = DEV[:512]. Activations: hidden states 0..24, mlp 14..23, post_operator_norm / post_ffn_norm outputs 12..23, o_proj inputs for attn layers 12..22.
- Result (original, DEV[:512]): acc 36.7%, mean logp(correct) -2.78. Pred letter counts A..J 94 95 78 58 40 54 38 28 15 12 vs gold 59 49 51 68 47 50 42 69 39 38.
  Acc by gold letter: A .58 B .53 C .47 D .34 E .34 F .42 G .29 H .22 I .21 J .24. Strong early-letter bias.
  Final mean centered letter logit (10-opt q): A +.98 B +1.51 C +.70 D +.18 E -.10 F +.87 G -.11 H -.67 I -1.82 J -1.55.
  Removing that constant offset (fit on half, test on other half; lens arithmetic) gives +1.5..+4.7 pts and +0.13..0.23 logp.
- Near misses: of 324 wrong, correct is 2nd in 77, 3rd 63, 4th 48 (spread out; not dominated by near misses).
- Logit lens (code/002): letter accuracy ~11% through layer 14, ~21-26% layers 15-21, 27.7% at h22, 35% h23, 36.9% h24. Mid layers lens-pick A almost always. Little "known in the middle then lost": at h22 26 q correct->final wrong vs 72 wrong->final right.
- DLA (code/003, onto centered letter unembeddings with final-norm scale): attn22 (full attn) is the retrieval head: correct-letter help +2.81. MLP22 suppresses it: help -1.19; per-question corr(attn22 push, mlp22 push) across letters median -0.88.
  MLP22 has one suppressor neuron per letter (C:1812 D:464 E:7357 F:6492 G:3344 H:397 I:2786 J:3310; none found for A,B), firing when that letter is retrieved. Removing them (lens counterfactual, code/005): acc +0.5..1.5 but logp -0.2..-0.8: not pursued.
- Early-letter bias sources (code/007, per-neuron DLA along the offset direction, layers 14-23): spread out but a set of neurons with near-constant activation and a smooth ordinal letter profile (early up, late down): L20 n1000 (+0.59), L22 n190 (+0.32), L17 n2401 (+0.28), n616, L23 n1047, n4597, n1903, L15 n4408, L19 n6612 ...
- code/008: selection rule mu>0.08 along offset, sd<mu, |correct help|<0.03 -> 16 neurons (work/sel_bias_neurons.json). Lens counterfactual removing their letter push: out-of-half logp +0.21..+0.26, acc +0.4..+2.7; all DEV512 acc 36.9->39.1, logp -2.78->-2.59.

## Edit E1 (work/edit_E1)
- code/009_make_edit.py work/sel_bias_neurons.json 1.0: for the 16 neurons, w2[:, j] -= P w2[:, j], P = projector onto centered letter readout subspace (E_letters * final-norm w * post_ffn_norm w of that layer). Tensors: w2 of layers 15,17,19,20,22,23 (rel change 0.12-0.9%).
  L15 [4408], L17 [2401,616], L19 [6612], L20 [1000,3296,3964,4353], L22 [190,1449,2820,4543], L23 [1047,4597,1903,3269].
## Pass 2: E1, both, DEV[512:768]
- RESULT pass 2 (E1 vs original, DEV[512:768], unseen by selection): acc 36.33 -> 39.06, paired +2.73 +- 1.51 (11 w->r, 4 r->w); logp(correct) +0.207 +- 0.047 (z 4.4). Lens predicted +0.19 logp: real network ~ linear DLA prediction.
  Pred counts edited A..J 30 43 39 22 23 21 30 20 15 13 (orig 32 53 50 28 22 18 19 14 8 12).
- code/010 per-head DLA: L22 head 12 is THE retrieval head (correct help +3.03, alone 34% acc); head 15 +0.48; heads 11,13,14 (and 0,8) anti-correct (-0.16..-0.23). L20 head 7 +0.13, L14 heads 12,15 small.
- code/011/012 lens screening: larger neuron sets or alpha>1 do not beat E1; removing L22 anti heads or boosting h12/h15 hurts logp (-0.2..-1.4). After E1 the full remaining constant offset removal adds nothing (half-split): E1 already captures the gain of an ideal per-letter debias.
## Pass 3: dose-response, E1 at alpha 0.5 (same 16 neurons, w2[:,j] -= 0.5 P w2[:,j]), both, DEV[512:768]
- RESULT pass 3 (E1 alpha 0.5, DEV[512:768]): acc +1.17 +- 1.17 (6 w->r, 3 r->w), logp +0.118 +- 0.024.
- code/013: on questions still wrong after E1 (lens), the wrong winner's margin comes from L22 head 12 (+4.97 mean, positive in 91%); MLP22 counteracts (-2.30). Head 12 alone is correct on only 6% of these vs 79% of the right ones: the remaining errors are retrieval of the wrong option (upstream belief), not late loss.
## Pass 4: E1 alpha 1.5 (overshoot: letter push reversed by 0.5), both, DEV[512:768]
- RESULT pass 4 (E1 alpha 1.5, DEV[512:768]): acc +2.73 +- 1.79 (14 w->r, 7 r->w, 56 preds changed), logp +0.255 +- 0.070.
  Dose-response on the same 256 DEV q (logp): alpha 0.5 +0.118, 1.0 +0.207, 1.5 +0.255; acc +1.17, +2.73, +2.73. Lens had predicted the optimum at 1.0; the real network still gains logp at 1.5 (more churn).
- code/014/015: head 12 itself has a non-ordinal letter preference (over-picks G and C, under-picks A/B), but equalizing its per-letter gain (lens) hurts badly: abandoned.
## Pass 5: E1 alpha 1.0 (work/edit_E1), both, HOLD[0:256]
- RESULT pass 5 (E1 a=1.0, HOLD[0:256]): acc 34.38 -> 34.77, +0.39 +- 1.52 (8 w->r, 7 r->w); logp +0.230 +- 0.043 (z 5.3). Pred counts orig A..J 46 55 36 33 18 27 16 7 7 11 -> 50 42 28 24 19 27 24 13 15 14 (gold 33 27 24 25 22 24 26 21 27 27).
## Pass 6: E1 alpha 1.5 (work/edit_E1a15), both, HOLD[0:256]
