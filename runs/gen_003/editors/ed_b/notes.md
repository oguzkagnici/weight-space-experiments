# Notes of editor ed_b, generation 3


## Setup
- Split (code/001): practice ids sorted, shuffled with random.Random(3003); HOLD = first 768, DEV = rest (2239). work/split.json.
- Mean rendered length 196 tokens; ~83% of questions have 10 options.

## Pass 1 (read, original, DEV[:240]) -- diagnostic, no edit
- returns topk50, L22 q_norm/k_norm/v_proj/gate_proj outputs at all positions, attn_last for L14/L20/L22.
- Goal: content side of L22.H12: which key features at option tokens carry the correct-option signal, how the H12 query matches them; where the gap between H12's attention argmax and the final answer comes from.
