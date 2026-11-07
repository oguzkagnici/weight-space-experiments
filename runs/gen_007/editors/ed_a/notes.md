# Notes of editor ed_a, generation 7

## Plan (start)
- Goal: an edit that is BITWISE silent off-target in llama.cpp Q8_0 (det1, REPACK/LLAMAFILE off ->
  ggml_vec_dot_q8_0_q8_0: activations entering w2 are quantised to Q8_0 per block of 32 neurons).
  => a neuron j contributes exactly 0 on a token iff round(h_j / (amax_block/127)) == 0, i.e. |h_j| < amax_block/254.
  A repurposed neuron is exactly silent on a token if the parent's h_j AND the edited h'_j both quantise to 0
  there (block amax then unchanged, other 31 values identical). The parent's own L20 n5414 is already such a gated
  neuron, so changes to n5414 act only where it is (or becomes) nonzero after quantisation.
- Targets considered: (1) commit moments n5414 misses (suggested start); (2) silent version of gen6 short-answer.
