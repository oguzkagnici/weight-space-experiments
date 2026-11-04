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

## Pass 2 (score original, same 512 DEV, acts L21-24 + MLP 21-23) -> fwd_02
- code/005: always-on, low-variance letter neurons do NOT carry the A/B preference (H1b refuted). L22 MLP total push to the correct letter is -1.35 (vs +3.1 from head 12).
- code/006 (H2 confirmed): L22 has one "leader suppressor" per letter (C n1812, D n464, E n7357, F n6492, G n3344, H n397, I n2786, J n3310) that fires (-3 to -4.6 logits on its letter) when its letter LEADS the stream entering the L22 MLP. No such neuron for A and B (A even has a booster, n2212). Total suppression of the leader (L22+L23): A -1.2, B -2.2, C -4.2, D -4.1, E -5.4, F -4.6, G -5.3, H -5.6, I -5.4, J -5.3. That asymmetry is the early-letter bias.
- code/007: suppression is ~accuracy-neutral overall: E-J leaders are right 39.1% vs final 38.1%; J 0.51 -> 0.40, H .32 -> .28, but C .28 -> .32, F .42 -> .47 (n ~ 35-90 per letter, noisy).
- code/008: ceiling of a pure letter-bias removal (cross-fitted constant offsets on recorded outputs) = +1.9 points (spread 1.8). Modest.
- code/009 (H3 refuted): no long-prompt collapse; within subjects, longer questions are easier (+7.8 points).
- Plan: equalise suppression by scaling the direct letter readout of the 8 L22 late-letter suppressors by alpha (code/010: removes (1 - alpha) of the column's component in the 9-dim letter-readout subspace). Screen alpha = 0.5, then 0 (and adding L23), on 256 fresh DEV questions (work/screen256.json; parent cached); confirm on HOLDOUT.
- Pass 3: alpha = 0.5 on L22 only (8 columns of model.layers.22.feed_forward.w2.weight), both, screen256.
- Pass 3 RESULT (alpha 0.5, L22 only, screen256): paired acc +1.56 (SE 0.95), +5/-1, 17 predictions changed; dlogp(correct) -0.141 (SE 0.024): accuracy up but the correct letter's log-prob down, so the suppressors also calibrate (they lower overconfidence). Edit saved in work/edit_a05.
- Pass 4: alpha 0.0 (full removal of the 8 L22 suppressors' letter readout), both, screen256.
- Pass 4 RESULT (alpha 0, L22 only): paired acc +1.56 (SE 1.23), +7/-3, 27 changed; dlogp -0.298 (SE 0.047). No gain beyond alpha 0.5 and noisier: alpha 0.5 is the better strength. Edit saved in work/edit_a0.
- Pass 5: alpha 0.5 on L22 AND the 8 L23 late-letter suppressors (C n1192, D n1876, E n2758, F n3520, G n1107, H n358, I n6512, J n1041), both, screen256.
- Pass 5 RESULT (alpha 0.5 L22 + L23): paired +1.17 (SE 1.41), +8/-5, 39 changed; dlogp -0.434. Adding L23 adds churn, not accuracy: keep L23 untouched.
- Pass 6: alpha 0.5 L22 suppressors + early-letter boosters' letter readout x0.5 (L22 n2212 A; L23 n1685 A, n2595 A, n1439 B), both, screen256.
