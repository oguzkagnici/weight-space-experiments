# Notes of editor claude, generation 1

## Setup
- DEV/HOLDOUT split of practice: work/split.json (seed 1234; HOLDOUT = 1,000 questions, DEV = 2,007). HOLDOUT is used only for confirmation.
- Pass 1 (planned): diagnostic score of the original on 512 DEV questions (37 per subject), with the residual stream after layers 8-24 and the per-head attention outputs (o_proj inputs, 16 heads x 128) of all 11 attention layers at the answer position.

## Pass 1 (score original, 512 DEV, acts) -> fwd_01. acc 38.1%, mean logp(correct) -2.77
- Predicted letters A..J: 81 80 88 55 46 39 41 31 27 24 vs true 62 59 47 63 43 48 38 44 50 58 -> early-letter preference.
- Logit lens (option-letter acc): L8 .13, L12 .12 (predicts "A" for 510/512 = token prior), L16 .21, L20 .26, L22 .29, L24 .38. The answer forms late.
- Head DLA (code/002): L22 H12 is THE answer mover: +3.10 logits to the correct letter on average (+0.29 on wrong questions). Other heads are small (|DLA| < 0.4). No head carries a strong letter-position slope.
- Bias source (code/003), mean centred letter logit on 10-option qs (A..J):
  - stream before L22: +1.18 +1.13 +.10 +.08 +.45 -.03 -.72 -.76 -.77 -.66
  - L22 attention: -1.10 -1.08 +.72 -.31 -.48 -.08 +1.12 +.52 -.28 +.96 (counteracts)
  - L22 MLP + L23 conv + L23 MLP: +.97 +1.23 -.11 +.40 -.05 +.83 -.63 -.41 -.47 -1.76 (re-inserts; J strongly down)
  - final: +1.05 +1.27 +.70 +.17 -.07 +.73 -.23 -.65 -1.53 -1.45; true answer shares are ~flat.
- On wrong qs: H12's own top letter = model's pick in 57%, = correct in 12%; H12 margin pick-correct +4.32; late blocks margin -1.97 (they lean to correct); stream before L22 +1.51.
- Hypothesis H1: a content-independent letter prior is written by specific late MLP neurons (L22/L23) at the answer position; removing that component (not a uniform shift) should improve accuracy, mostly on late-letter answers.
- Pass 2 (planned): same 512 DEV qs, original, acts: hidden 21-24, MLP acts of layers 21, 22, 23, conv out_proj input of L23.
