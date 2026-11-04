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

## Pass 2 result: L22 suppressor bank x0.5, S=DEV[512:768]
- paired acc +0.78 +- 0.78 (3 w->r, 1 r->w, 13 preds changed); logp_correct -0.052 +- 0.014. Letter bias essentially unchanged.
- code/009: the edit just raised the originally-top letter by +0.41 (vs direct estimate ~1.5): strong downstream compensation (L23), and it acts like a sharpening knob, not a debias. ABANDONED.
- Lesson: direct-effect estimates for L22 edits overestimate; L23 edits have no downstream layers.
- S original: acc 32.8, bias [1.15 1.31 1.2 .28 .22 1.2 -.25 -.92 -2.59 -1.59] (same pattern as DEV[:512]).

## Pass 3 (planned): EDIT = project the centered letter-unembedding subspace (9-dim, span of (wn*E_k - mean)) out of the w2 columns (in post_ffn_norm-scaled space) of 12 near-constant bias neurons: (20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820). code/008 args "1.0 12". Both on S with acts h20-24, post_ffn 20-23, post_op 21-23.

## Pass 3 result: EDIT v1 (12 bias neurons, letter-subspace projected out), S
- paired acc +2.34 +- 1.35 (9 w->r, 3 r->w, 34 changed); logp_correct +0.190 +- 0.041 (z 4.7). Pred letters A35 B37 C36 D25 E17 F25 G26 H21 I22 J12 (orig A39 B45 C45 D28 E20 F27 G18 H13 I12 J9).
- Bias on S: orig [1.15 1.31 1.2 .28 .22 1.2 -.25 -.92 -2.59 -1.59] -> v1 [.64 .35 -.17 -1.08 -.08 1.09 .02 .09 -.34 -.52]. Direct-effect prediction from pass-1 acts was [.4 .3 -.6 -1.1 -.2 1.2 .5 .1 -.3 -.4] => essentially NO self-repair for this edit (unlike the L22 suppressor edit).
- Saved as work/best_v1.
- code/013: blanket "all low-cv neurons" variants not better than hand-picked 12. code/016 per-head: L22h12 is the answer-copy head (content +2.96 logits on correct letter) and has its own bias (G +1.6, B -1.4); L22h6, L20h7 small near-constant bias heads.
- code/017 dose (direct estimate on pass-1 acts): strength .25/.5/.75/1/1.25 -> acc 35.7/36.5/37.1/36.9/36.5, logp -2.75/-2.69/-2.66/-2.66/-2.67 (peak ~0.75-1.0: smooth dose-response).

## Pass 4 (planned): EDIT v2 = v1 + 15 more neurons [(22,4543),(21,218),(23,6518),(23,4531),(22,5653),(23,514),(23,3073),(23,2189),(21,4481),(22,2212),(22,3998),(23,1439),(23,7466),(21,3455),(21,4902)] + head L22h6 (o_proj cols 768:896), all letter-subspace projected (code/018). Both on S. (direct estimate: acc 37.5, logp -2.57 vs v1 36.9/-2.66 on DEV[:512])

## Pass 4 result: EDIT v2 on S
- paired acc +3.12 +- 1.55 (12 w->r, 4 r->w, 42 changed); logp_correct +0.247 +- 0.047. Pred letters A36 B29 C34 D33 E25 F24 G19 H19 I24 J13. Better than v1 on same S (+2.34 / +0.190). Saved work/best_v2.

## Pass 5 (planned): EDIT v2 (unchanged), both on HOLDOUT[0:256]. Pass 6: v2 on HOLDOUT[256:512].
- Pass 5 result (v2, HOLDOUT[0:256]): acc 36.72 -> 38.67, paired +1.95 +- 1.61 (11 w->r, 6 r->w); logp_correct +0.188 +- 0.044. Pred I 11->24, H 12->18, B 36->26.
- Pass 6 result (v2, HOLDOUT[256:512]): acc 34.38 -> 35.94, paired +1.56 +- 1.56 (10 w->r, 6 r->w); logp +0.168 +- 0.043.
- HOLDOUT[0:512] combined v2: acc +1.76 +- ~1.12 ; logp +0.178 +- ~0.031.
- code/019: on S+HOLDOUT (768 Qs) the residual bias of v2 is [.3 -.1 -.13 -.13 .01 .78 -.26 -.06 -.04 -.37] (orig [1.25 1.34 .99 .21 .14 1.28 -.2 -.88 -2.33 -1.8]). Perfect constant debias on top of v2 would give only +0.8pp more (CV): v2 captures ~75% of the bias-removal ceiling.
- code/020: further bias neurons/heads add ~nothing in direct estimate. Bias route saturated.

## Pass 7 (planned): EDIT v2 unchanged. Transfer check on all practice questions with <=6 options (195 Qs; 153 have <=4 options), both. GPQA has 4 options.
- Pass 7 result (v2, all 195 practice Qs with <=6 options; 142 not used in DEV[:768]): acc 43.59 -> 45.13, paired +1.54 +- 0.88 (3 w->r, 0 r->w); logp -0.052 +- 0.030 (slightly worse). 4-option subset (151): acc 45.7 -> 47.7; A-D bias [.64 .34 .33 -1.31] -> [.58 -.05 .2 -.74] (D under-pick halved). Transfers in accuracy; logp mixed.

## Pass 8 (planned): EDIT v3 = v2 + L22 suppressor bank equalized: w2 columns scaled 1812 x.81, 464 x.42, 7357 x.33, 6492 x.52, 3344 x.50, 397 x.66, 2786 x.48, 3310 x.53 (A389, B6620/5859 untouched) (code/022). Both on S, compare to v2 on S (+3.12 / logp +.247).
- Pass 8 result (v3 on S): +2.73 +- 1.60 (12 w->r, 5 r->w), logp +0.197 +- 0.050. WORSE than v2 on S (+3.12 / +0.247). Equalizing the L22 suppressor bank does not help; v3 dropped. work/best_v3 kept for reference.

## Pass 9 (planned): EDIT v2 (re-made with code/018), both on HOLDOUT[512:768].
- Pass 9 result (v2, HOLDOUT[512:768]): acc 35.94 -> 37.50, paired +1.56 +- 1.83 (13 w->r, 9 r->w); logp +0.193 +- 0.046.
- v2 on HOLDOUT[0:768] (3 passes pooled): acc +1.69 +- ~0.96 pp ; logp_correct +0.183 +- ~0.026.

## Pass 10 (planned): EDIT v1 (code/008 "1.0 12"), both on HOLDOUT[0:256] (same Qs as pass 5 for v2: +1.95).
