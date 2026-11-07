# Notes of editor ed_a, generation 7

## Plan (start)
- Goal: an edit that is BITWISE silent off-target in llama.cpp Q8_0 (det1, REPACK/LLAMAFILE off ->
  ggml_vec_dot_q8_0_q8_0: activations entering w2 are quantised to Q8_0 per block of 32 neurons).
  => a neuron j contributes exactly 0 on a token iff round(h_j / (amax_block/127)) == 0, i.e. |h_j| < amax_block/254.
  A repurposed neuron is exactly silent on a token if the parent's h_j AND the edited h'_j both quantise to 0
  there (block amax then unchanged, other 31 values identical). The parent's own L20 n5414 is already such a gated
  neuron, so changes to n5414 act only where it is (or becomes) nonzero after quantisation.
- Targets considered: (1) commit moments n5414 misses (suggested start); (2) silent version of gen6 short-answer.

## Findings before pass 1 (code/001-003)
- Bank: 53 cut; 5 of them closed the think and were cut in the visible answer (1740 D ok, 9274 B ok, 8198 C ok,
  6829 F wrong, 6014 no stated answer). Mid-trace "answer is" statements in cut traces are mostly hedged two-option
  lines ("Thus likely answer is G or D"); firm+correct mid statements: 8932 only (+6941 "likely intended J").
- code/003: no MLP neuron in any layer has a Q8_0 w2 column that quantises to zero (min 1934 of 2048 nonzero).
  So exact silence must come from ACTIVATIONS quantising to 0 (|h_j| < block amax/254).

## Pass 1 (read, original = parent), work/req1.json
- 15 bank traces (code/004: 6 targets 1740 9274 8198 6829 6014 8932 + 9 finished, one per category), 35,806 tokens.
- return token_logprobs, mlp [20,21,22,23] all positions, inputs of w1 at L20, L22, L23.
- Purpose: (1) per-neuron fraction of tokens with nonzero Q8 activation (quiet-neuron screen), (2) gate design at </think>.
- code/005: finished bank traces: visible-answer letter == think's last stated letter in 41/43 (7811 no stated
  letter but correct C; 5603 no answer). Tokens: '"The'=14225 ' answer'=5141 ')".'=94078 <|im_end|>=124900 </think>=124902.

## Plan A: exactly-silent short visible answer (L23 hosts)
- An L23 neuron changes only the logits at its own position (nothing reads L23's output later), so teacher-forced
  parent reads of '...</think>"The answer is (X)".<|im_end|>' give exactly the inputs the edited model sees.
- Need 3 L23 neurons whose parent activations quantise to 0 on all thinking tokens, and gates that are <<0 (silu~0)
  off-target: S1 at </think> -> '"The'; S2 at '"The' after </think> -> ' answer'; S3 at ')".' after it -> <|im_end|>.

## Pass 2 (read, original), work/req2.json (code/006), queued while pass 1 waits
- 46 synthetic texts (prompt+think+'</think>"The answer is (X)".<|im_end|>', X = think's last letter; 41 finished +
  closed-cut) + 6 cut traces; 92,954 tokens. return token_logprobs, topk 10, mlp [23], input of L23 w1, all positions.
