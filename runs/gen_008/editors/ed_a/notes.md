# Notes of editor ed_a, generation 8


## Diagnosis (no passes)
- code/002: option sweeps ("Let's examine each option", "Option A:" restarts, >300 chars apart). Finished traces: mean 1.26 sweeps; cut: 2.89.
  Cut traces with >=2 sweeps: 33/53. Finished with >=2: 13/43 (8 correct).
- code/003: crude regex for stated letters in cut traces: 35/53 mention some "answer (X)"; first-stated letter = gold 11, last 13 (noisy regex).

## Pass 1 (read, original) -- plan
- texts: trace_text of 80 bank traces (all 43 finished + 37 cut, incl. all 33 cut with >=2 sweeps), 193,736 tokens.
- return: token_logprobs, topk 20, inputs of L23 and L22 feed_forward.w1 (= ffn_norm output), positions all.
- goal: fit a gate for "paragraph break right before a 2nd/3rd option sweep" at the L23 MLP input; see the next-token dist there.
