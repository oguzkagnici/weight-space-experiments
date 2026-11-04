# Notes of editor ed_a, generation 2


## Setup
- Split (code/001): practice ids sorted, shuffled with Random(20262); DEV = first 1500, HOLDOUT = remaining 1507 (work/split.json).
- Starting point: L22 head 12 option selection (suggested). Hypothesis: non-content factors (option position / recency / letter) steer H12's attention.

## Pass 1 (read, original, DEV[:512]): diagnostics
- Edit: none. Return: hidden 12..24 last, mlp 21-23, L22 o_proj input, L22 gate_proj output, attn_last [10,14,18,20,22].
- Pass-1 findings (DEV[:512], parent 34.4% by lens, 10-opt pred A..J 75 78 62 53 39 39 41 27 16 19 vs gold flat ~45):
  - Logit lens: h20 26%, h21 26%, h22 28%, h23 34%, h24 34%; up to h21 the stream mostly predicts A/B (prior).
  - L22 heads (0,1,3,8,10,11,12,13,14,15) all put more attention mass on early option spans: L22h12 share by slot A .22 B .16 C .11 ... I .05 J .08; argmax option A in 236/512.
  - code/006: per-token log-attention regression (per-question demeaned): slot coefficients vs A for h12: B -.83 C -1.16 D -1.33 E -1.77 F -1.99 G -2.14 H -2.23 I -2.31 J -2.07; token distance only -0.15 per 100 tokens. => slot identity, not RoPE recency.
  - code/005: q at answer position barely reads the letter-unembedding subspace (7% of norm, weights read it 0.6x random) -> query letter-prior feedback via unembedding dirs is NOT the mechanism.
## Pass 2 (read, original, 120 10-option DEV[:512] questions, positions all): L22 q_norm/k_norm/q_proj/k_proj outputs + attn_last 22
- Pass-2 findings (code/008-011): L22 attention scores = q.k/sqrt(128) on recorded q_norm/k_norm outputs with NO rotary needed (resid var 0.3%; NoPE-like).
  - Slot-mean keys (per kv group) have a dominant ordinal PC1 v1 (split-half r .97): slot scores A -4.3 ... J +2.0. Rank 1 explains the whole slot preference (h12 range 2.06 of 2.22 nats).
  - q at answer position has a large ~constant component along v1 (q.v1 ~ -2.6..-3.9, ~32% of |q|; sd of slot effect over questions only .17) => prefers low k.v1 = early tokens.
  - k.v1 is a general position-like signal: question text -4.65, slots A..J -5.66 -3.77 -2.84 -1.89 -1.15 -1.19 -.6 -.3 .19 .31, template tail -3.65. Within options k.v1 ~ .53*slot + .36*abspos/100.
  - L22h12 mass: question text .41, options .43 total, tail .15.
## Pass 3 (both, DEV[512:768] = S1): edit E1 alpha=1.0
- Edit: model.layers.22.self_attn.q_proj.weight, all 16 heads: W_h -= alpha * t t^T W_h, t = normalize(q_norm.weight * v1_g) (v1_g = PC1 of slot-mean keys, kv group g=h//4). code/012.
- Pass 3 result (S1 = DEV[512:768], n=256): acc 38.28 -> 40.23, paired +1.95 (SE 1.79), 13 w->r / 8 r->w; logp(correct) +0.165 (SE 0.056).
  - pred A..J 44 40 44 30 25 22 22 15 9 5 -> 22 27 32 32 28 28 38 19 19 11 (gold 36 22 22 28 29 24 25 27 17 26).
  - centred letter logp (10-opt) orig [1.19 1.43 .68 .09 .34 1.08 .1 -.94 -2.27 -1.7] -> [-.38 .24 -.17 -.2 .28 .95 .5 -.07 -.85 -.31].
  - code/013: cross-fitted constant-letter debias of BOTH models: orig .371 vs edit .394 (10-opt, n=213) -> gain is not only a letter offset.
  - saved as work/edit_E1_a1.0
## Pass 4 (both, S1): E1 alpha=0.5 (dose-response), same tensor/heads.
- Pass 4 result (S1): E1 alpha=0.5: paired +1.95 (SE 1.03), 6 w->r / 1 r->w; logp +0.129 (SE 0.031). Centred bias [.38 .9 .33 -.04 .36 1.07 .29 -.58 -1.69 -1.01]. Debiased-acc orig .371 edit .379.
  - Dose-response logp: alpha .5 -> +.129, alpha 1 -> +.165. Keep alpha=1 for L22 (principled full removal).
## Pass 5 (read, original, same 120 questions as pass 2): L20 q_norm/k_norm all positions + attn_last 20 -> does sliding-window L20 have the same primacy mechanism?
- Pass-5 findings (code/014-015): L20 uses RoPE (half-rotation, theta 1e4, full head dim; fit resid 0.000; NoPE 0.11). Slot-mean keys again have a reliable PC1 (split-half r .95-.96) living mostly in low-frequency RoPE pairs (kv1: 67% in pairs 48-63), so rotated slot scores ~= pre-rope ones. L20 slot preference is "A first": e.g. L20h1 A +1.02, B +.13, rest ~-0.1..-0.3; h12/h13/h15 A +1, B +.3-.5. q fraction along v1 .2-.37.
## Pass 6 (both, S1): E1 (L22 alpha=1, all heads) + E2 (L20 q_proj, same construction with L20 slot-key PC1, alpha=1, all 16 heads; code/016)
