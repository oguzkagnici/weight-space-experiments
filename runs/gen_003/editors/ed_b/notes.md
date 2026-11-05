# Notes of editor ed_b, generation 3


## Setup
- Split (code/001): practice ids sorted, shuffled with random.Random(3003); HOLD = first 768, DEV = rest (2239). work/split.json.
- Mean rendered length 196 tokens; ~83% of questions have 10 options.

## Pass 1 (read, original, DEV[:240]) -- diagnostic, no edit
- returns topk50, L22 q_norm/k_norm/v_proj/gate_proj outputs at all positions, attn_last for L14/L20/L22.
- Goal: content side of L22.H12: which key features at option tokens carry the correct-option signal, how the H12 query matches them; where the gap between H12's attention argmax and the final answer comes from.

## Pass 1 results (DEV[:240], original): the content side of L22.H12 is saturated
- Final acc .396 (lens from topk). H12 option-attention-mass argmax .392; agree with final 80%; 11 rescued / 12 lost downstream. No gap between H12's pick and the final answer to recover. (code/003)
- q.k from recorded q_norm/k_norm reproduces H12 attention (max |dp| .036). H12 attention: newline after option .44, last content token .23, letter token .12, other content .06, outside options .17. (code/004)
- Adding the mean gold-minus-wrong key direction d to the H12 query HURTS in 2-fold CV (beta 2..16: .396 -> .375 .. .30). q already has q.d_hat = 4.6 of |q| 12.2. (code/004)
- Query = near-constant part (cos .93) + question-specific dq. Constant query alone .342; actual .396 -> dq is worth ~5 pts. Amplifying dq (gamma 1.5/2/3): .404/.400/.388 = no gain. (code/005)
- Slot (position) bias of H12 option scores: A,B,C +.4 vs rest; CV per-slot correction: .374 -> .364 (no gain). Projecting the slot-mean key subspace out of q (rank 1..9) over-corrects (A never picked) and loses .01-.09. (code/006)
- Temperature: T=2 on H12 logits gives .417 (8 w->r, 2 r->w), T=0.5 gives .333. But per-head temperature can't be edited cleanly: q_norm/k_norm weights are shared across heads, the eps regime does not apply (rms(q/g) ~ .94-1.05), and only ~17% of |q|^2 lies in the key null space (pure T gain 1.10 -> +2/-0). (code/007-011)
- Length-normalizing H12 option mass: no change (attention too peaked). Keys at option ends grow with option length (slope 1.3), gold is also longer (code/012). 
- Removing query directions (key-covariance eigenbasis) with negative mean contribution to gold-vs-wrong margin: CV 13/13 -> nothing (code/013).
- H12 value letter code at option ends decodes the right letter 97-99.6% at all option lengths (code/014).
- Other L22 heads select options about as well (H14 .404, H15 .396) but agree with H12 81-88%; best combined vote +5 net of 240 (code/010, 015).
=> Conclusion: given the keys, the H12 readout is close to optimal; errors live in the option-end beliefs (keys), as gen-2 ed_b found. Abandoning the suggested starting point for edits.

## Pass 2 (score, original, DEV2 = dev[240:752], 512 q) -- diagnostic, no edit
- activations at answer position: h20-24, mlp20-23, L22 o_proj input (heads), post-norm inputs (L20-23), L22 gate. Purpose: exact DLA of the letter stage (L22 heads, MLP neurons) to design a letter-stage edit.

## Pass 2 results (DEV2, 512 q, original): letter stage
- acc 35.16; preds A82 B93 C98 D54 E41 F44 G36 H26 I21 J17 vs gold 64 63 54 70 39 47 51 45 42 37.
- Exact DLA (code/016), centered 10-opt mean letter logits: h22 [1.14 1.16 .18 .09 .45 -.04 -.70 -.79 -.81 -.69]; attn22 [-1.07 -1.08 .96 -.31 -.64 -.02 1.26 .35 -.36 .91] (gold-help 2.79; H12 alone 2.86 with its own tilt A -.59 B -1.41 G +1.18); mlp22 [.8 .78 .23 .4 ...-.89] (gold-help -1.14); L23 [.13 .44 -.43 .02 .18 .98 ...-.83] (F +.98 comes from layer 23). Stage acc: h22 .268, +attn22 .361, +mlp22 .355, final .350.
- KEY: cross-fitted constant per-letter debias at the final logits gains +0.1 pt (sd .7 over 20 splits) on the 424 10-option DEV2 questions (code/017). The letter bias is real but costs (almost) no accuracy here; consistent with gens 1-2 letter fixes ~0 on hidden.
- No off-by-one (neighbour) error excess (code/019). No format anomaly by category (code/018).

## Pass 3 (read, original, DEV[:120], residual h10..h22 all positions)
- "Key lens": H12's option pick if its keys were read from earlier residual layers (operator_norm/k_proj/k_norm of L22 applied to h_l, same query). Tests whether a middle layer at the option tokens holds a better belief than h22 (knows-then-loses at option positions). Linear readout analysis only.

## Pass 3 results (key lens, DEV[:120])
- H12 gold pick with keys read from h10/12/14/16/18/20/22: .158 .200 .275 .375 .383 .383 .425. Monotone build-up, no "knows then loses" at option positions (h20 vs h22: right only at h20 3, only at h22 8). (code/021)
- Scaling single blocks' contributions in the key lens: all within +-3 of 120 (code/022). The belief is distributed.

## New finding: H12 attends to content-free option letter tokens
- H12 puts .116 of attention on "(X" letter tokens. These precede the option text (causal), so their keys have no option content. Dropping them from H12's option vote: recomputed q.k lens +8/-0 (B over-pick 36 -> 26), recorded attention +5/-2 (code/023, 024). Many L22 heads gain from dropping letter tokens (H3 +25/-5, H0 +11/-3, H11 +14/-4), but only H12 (gold-help 2.86) and H15 (.43) copy letters positively; H15 lens neutral (code/025).
- Edit (code/027): u = mean over DEV[:240] of (mean letter-token key - mean option end/newline key), group 3 keys; w = normalize(q_norm.weight * u); head-12 rows of model.layers.22.self_attn.q_proj.weight: W_h <- W_h - alpha w (w^T W_h). Rank-1, head 12 only. q_norm renormalizes.
- Lens (2-fold CV): alpha .5/1/1.5/2: +6/-1, +8/-3, +10/-4, +9/-6; letter mass .071/.043/.026/.016.

## Pass 4 (score both, DEV3 = dev[752:1008], 256 q): edit = letterproj alpha 1.5, head 12 (only tensor: model.layers.22.self_attn.q_proj.weight)
- Pass 4 result: parent 27.73 on DEV3 (hard sample). Edited 28.91: paired +1.17 (SE .87), 4 w->r / 1 r->w, 14 preds changed; dlogp(correct) +.022 (SE .023); gold rank -.055 (SE .037); A preds 44 -> 38, H 17 -> 21. Small, consistent sign. Saved as work/best_1_letterproj_h12_a15.

## Pass 5 (score both, DEV3 again, 256 q): edit = letterproj on ALL 16 L22 heads (per-kv-group direction u_g, w_g = normalize(q_norm.weight*u_g)), alpha 1.0, except H12 alpha 1.5 (code/028). Tensor: model.layers.22.self_attn.q_proj.weight only.
- Pass 5 result (all heads a1.0, H12 a1.5): acc +0.39 (SE 1.17), 5/4, 20 preds changed; dlogp +.040 (SE .023); gold rank -.078 (SE .039). Better continuous signals than H12-only, accuracy noisier. Saved work/best_2_letterproj_all_a10.

## Pass 6 (score both, DEV3, 256 q): all 16 L22 heads alpha 1.5 (dose step). Tensor: model.layers.22.self_attn.q_proj.weight.
- Pass 6 result (all heads a1.5): acc 0.00 (SE .96), 3/3, 19 changed; dlogp +.065 (SE .023); rank -.055 (SE .037). Saved work/best_3_letterproj_all_a15.
- DEV3 summary: the letter-token projection gives a consistent small gold-rank improvement (-.055 to -.078) and a dose-dependent logp gain (+.022/+.040/+.065); accuracy changes are within noise (+1.17/+.39/0).
- CHOICE (made on DEV3 only, before any HOLDOUT): candidate 1 = H12-only alpha 1.5 (most targeted, the only head whose lens justifies it, best DEV acc); candidate 2 = all heads alpha 1.0 / H12 1.5 (best DEV gold rank). Each measured once on HOLDOUT[0:512] (two 256 passes each).

## Pass 7 (score both, HOLDOUT[0:256]): candidate 1 = H12-only letterproj alpha 1.5 (q_proj L22 rows 1536:1664)
- Pass 7 result: acc -0.78 (SE 1.11), 3/5, 21 changed; dlogp +.046 (SE .025); rank +.016 (SE .040).

## Pass 8 (score both, HOLDOUT[256:512]): candidate 1 (same edit)
- Pass 8 result: acc +0.78 (SE .96), 4/2, 12 changed; dlogp -.006 (SE .023); rank -.066 (SE .037).
- Candidate 1 HOLDOUT[0:512] combined: acc 0.00 (SE ~.73), 7 w->r / 7 r->w; dlogp +.020 (SE ~.017); gold rank -.025 (SE ~.027). Null.

## Pass 9 (score both, HOLDOUT[0:256]): candidate 2 = all L22 heads letterproj alpha 1.0, H12 alpha 1.5 (work/best_2)
- Pass 9 result: acc 0.00 (SE 1.11), 4/4, 25 changed; dlogp +.070 (SE .025); rank -.020 (SE .040).

## Pass 10 (score both, HOLDOUT[256:512]): candidate 2 (same edit)
- Pass 10 result: acc 0.00 (SE .96), 3/3, 13 changed; dlogp +.024 (SE .022); rank -.043 (SE .038).
- Candidate 2 HOLDOUT[0:512] combined: acc 0.00 (SE ~.74), 7 w->r / 7 r->w; dlogp +.047 (SE ~.017, z~2.8); gold rank -.031 (SE ~.027).

## Submitted (code/029)
- letterproj_L22_all (candidate 2 above; HOLDOUT 512: acc 0.00, dlogp +.047) and letterproj_L22_h12 (candidate 1; HOLDOUT 512: acc 0.00, dlogp +.020). edited/ currently = H12-only edit.
- Report: report.md. Budget: 10/10 passes used.

## Submitted (code/029)
- letterproj_L22_all (candidate 2; HOLDOUT 512: acc 0.00, dlogp +.047) and letterproj_L22_h12 (candidate 1; HOLDOUT 512: acc 0.00, dlogp +.020). edited/ currently = H12-only edit.
- Report in report.md. Budget: 10 of 10 passes used.
