# Notes of editor claude, generation 7


## Process notes
- As orchestrator I saw generation-6 selection results per question id (gain/loss pattern only, not content); 6 of those ids are in this generation's selection set. I never use selection ids here.
- I wrote generation 6's `finalist_commit_L21`; its report is in the history. I do not read my old gen-6 folder; data is re-derived here.

## Plan
- Goal: an edit that is exactly silent off-target, then the two-finalist commit (and possibly the churn -> Hence idea) on it.
- Key hypothesis (H1): llama.cpp quantises the MLP activation vector h (7,680) to Q8_0 in blocks of 32 before the down projection (q = round(h/d), d = block absmax/127). A neuron j with |h_j| < absmax/254 on every token contributes exactly 0 in the parent. Repurposing such a neuron, with a gate whose off-target |h_j| also stays below that bound, leaves the down-projection output bit-identical off-target, so det1 traces stay byte-identical until the gate fires.
  - w1/w3 rows are quantised per row (blocks along the 2,048 input dim), so a new row does not touch other neurons. The w2 column is clipped below each row block's absmax (gen 5), so other neurons' w2 quantisation is unchanged.
- Pass 1 (read, original): 4 bank traces, first ~1,200 generated tokens, `mlp` layers 12-23, positions all. Per neuron: share of tokens where |h_j| >= absmax(block)/254 (would be nonzero in Q8), and the max ratio.
