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
