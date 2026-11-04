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
