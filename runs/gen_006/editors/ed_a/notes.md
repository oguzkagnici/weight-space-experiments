# Notes of editor ed_a, generation 6


## Diagnosis on the trace bank (code/001-005)
- Cut traces with an answer statement inside the think: 23 of 53 (broad regex). First statement correct 12/23; hedged 16 (8 correct), firm 7 (4 correct).
  Mid-trace (<0.75 of trace) first statements: 11, only 4 correct (8262, 6941, 8932, 5325). Late (>0.85): 11, 8 correct.
- NEW FAILURE FOUND: 5 of 53 cut traces CLOSED the thinking (</think>) but were cut while writing the visible answer
  (1740 D*, 9274 B*, 8198 C*, 6014 D*, 6829 F): think 2779-3054 tokens, the post-think content re-derives the whole
  solution (finished traces: content 248 tokens mean, 61-405). 4 of these 5 had the correct answer settled in the think.
- In 42 of 43 finished traces the content's answer == the last answer stated in the think (1 exception: content gave no answer).
  => the post-think content is a redundant re-derivation; when the think ends late it costs correct answers.
- The content starts right after the </think> token (no newline). First words: To 17, The 16, **Step 7, When 3, We 2 ...
- Plan: a gated neuron that fires only at the </think> position / first content tokens and makes the answer start with
  "The answer is (X)." (copying the think's settled letter) so the content is short. Thinking untouched.
- Exact silence idea: llama.cpp quantises the MLP activation vector to Q8_0 (blocks of 32 neurons) before w2. A neuron whose
  |activation| < blockmax/254 at every thinking token contributes exactly 0 there; repurposing such a neuron (new gate
  far below 0 in thinking) leaves thinking bit-identical. No w2 column quantises to zero in weights (code/006).

## Pass 1 plan (read, original): work/req_p1.json (code/007)
13 bank traces (5 closed-but-cut + 8 finished) natural text, and 13 variants = think + "</think>The answer is (X)."
Return token_logprobs, topk 20, mlp[23], inputs L23 w1, all positions (63k tokens). No edit staged.
