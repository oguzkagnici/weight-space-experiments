# Notes of editor ed_a, generation 1


## Setup
- code/001: split practice ids (sorted, shuffled seed 0) -> work/split.json: DEV = first 2000, HOLDOUT = last 1007.
- Practice answer letters: A 354, B 330, C 309, D 356, E 270, F 297, G 280, H 281, I 292, J 238 (true prior is front-loaded).

## Pass 1 (planned): score original on DEV[:512], acts: hidden 0..24, mlp 12..23, o_proj inputs (attn layers), post_operator_norm / post_ffn_norm outputs (all layers). No edit.

## Pass 1 results (original, DEV[:512])
- acc 33.6%. pred letters A85 B99 C76 D33 E48 F67 G47 H23 I20 J14 vs true A69 B48 C51 D48 E53 F55 G46 H52 I41 J49. Strong early-letter (A,B,C,F) preference, H/I/J starved.
- Mean centered logit bias (10-opt Qs): [1.15 1.66 .57 .16 .03 1.27 -.05 -1.14 -2.09 -1.58]. Offline (2-fold CV) subtracting it: 33.6 -> 37.5% (ceiling for a pure bias fix).
- Wrong answers: correct letter ranked 2nd in 85/340, 3rd in 62 -> many near misses.
- Logit lens: letters become readable only at h23/h24 (acc 13% at h12, 24% h20, 33% h23).
- Direct logit attribution (code/002,004): L22 attention (post_operator_norm out) is THE answer-letter writer (+2.84 logits on correct letter). L22 MLP is -1.13 on correct letter, -2.87 on the letter L22-attention retrieved.
- code/005-006: L22 MLP has one "retrieved-letter suppressor" neuron per letter: A389 B6620(+5859) C1812 D464 E7357 F6492 G3344 H397 I2786 J3310. Each fires only when its letter is the L22-attn top candidate and pushes it down by: A -1.5, B -0.9/-0.55, C -1.9, D -3.7, E -4.7, F -3.0, G -3.1, H -2.4, I -3.2, J -2.9 logits. => A/B are barely suppressed, late letters strongly: matches the A/B preference. L23 has a second, weaker and more even, bank (A3631 B7209 C1192 D1876 E2758 F3520 G1107 H358 I6512 J1041, ~-1 to -1.9).
- Direct-effect offline estimate (final rms fixed, no downstream): L22 bank x0.5 -> 36.7% (+3.1); x0 -> 35.9%. Mean logp correct drops (more confident).
- code/007: near-constant bias neurons (cv<0.6) writing early-letter bias: L20 n1000 (0.57 proj), L22 n190, L17 n2401, L17 n616, L23 n1047, ... Removing letter-subspace of top 5: offline +2.5pp alone, +3.3 combined with sup x0.5.

## Pass 2 (planned): EDIT = L22 w2 columns [389,5859,6620,1812,464,7357,6492,3344,397,2786,3310] x0.5 (code/008 args 0.5 0). Both, DEV[512:768] (screening set S).
