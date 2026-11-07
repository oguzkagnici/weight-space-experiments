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

## Pass 1 results (read, original) -> work/fwd_01 (code/008-018)
- At </think> the first answer token is "The" with P .55-.99 in 9/13 (others To/When/**).
- After "</think>The", P(" answer") = e^-6 .. e^-19 (median e^-15): the parent never starts its answer with "The answer".
  After a forced "The answer is (", the letter copies the think's statement with P=1.00; after "(X)." P(<|im_end|>) only
  .01-.03 (one trace .99): it goes on to a full explanation ("  \n\n" .4-.9).
- No L23 neuron is Q8-silent (every neuron is the block max somewhere; nonzero at ~90% of thinking tokens); no silent
  input dims either. Neuron "loudness" (rms act x logit spread) varies only 3x (quietest 4526, 810, 7602, 135, 2759; n74 also quiet).
  => any repurposed neuron perturbs thinking a little (as in gen 5). L23 MLP edits are position-local (no later layer).
- Gates at L23 MLP input (PC-cleaned mean-diff, top-200 thinking PCs projected out, bias proxy = PC-cleaned mean, CV .095):
  p0 (</think>): held-out thinking max z -3..-4.5, held-out targets min 12-17 => clean.
  p6 (after "The answer is (X)."): held-out thinking max -7..-9, targets min 6.6-20; natural answer text max ~4.7.
  p1 ("The" right after </think>): NOT separable at L23 (targets min z ~7 vs thinking max ~8, content ~14).
- Decision: explore openers whose continuation to "answer is (X)" is natural. Note '"The' is ONE token; thinking traces
  end with 'Hence final output: "The answer is (X)".' in most finished cases (induction candidate).

## Pass 2 plan (read, original): work/req_p2.json (code/020)
10 bank traces x 6 openers after </think>: '"The answer is (X)".', 'Therefore, the answer is (X).', 'The correct answer is (X).',
'**The answer is (X).**', 'Thus, the answer is (X).', 'The final answer is (X).'. token_logprobs, topk20, L22+L23 w1 inputs, all positions.
No edit staged yet (edited/ has a v1 test build from code/015 -- not used; pass is on the original model).

## Pass 2 results (read, original) -> work/fwd_02 (code/021)
Forced logprob of each opener token after </think> (10 traces), and P(<|im_end|>) after it:
- '"The answer is (X)".' : first token '"The' lp -9.4..-16 (needs a push); then ' answer is (X)".' natural (sum lp -1.8..-8.7, median ~-3,
  copied from the think's own last line 'Hence final output: "The answer is (X)".'); P(eos) after: .98-1.00 in 5/10, .37-.47 in 2, .05-.19 in 3.
- 'Therefore/Thus, the answer is (X).': ' answer' after ', the' lp -4..-17; P(eos) ~0. 'The correct/final answer': lp -9..-21; P(eos) ~0.
- '**The answer is (X).**': lp ~-17 at 'The'. => choose '"The'.

## Edit v2a (code/022, cfg work/edit_cfg2_v2a.json, snapshot work/snap_v2a) -- current edit
L23 MLP, two repurposed quiet neurons (w1 row, w3 row, w2 column each):
- n4526: gate = PC-cleaned (200 thinking PCs) mean-diff direction </think>-position minus thinking mean, threshold z*=15 thinking-sd
  via PC-cleaned mean proxy; alpha 5 (g at worst thinking token -13, at think-final '".' lines -57, targets >= +71);
  up = proxy (u~4); w2 col = (E['"The'] - weighted E[The,To,When,**,We]) x final-norm gain x post_ffn_norm gain, clipped to
  0.98 x Q8 block absmax of the other 31 neurons (no block scale changes).
- n810: gate = same construction for the position right after '"The answer is (X)".' (p6, 6 tokens after </think>), z*=22,
  alpha 4 (thinking max -32; hard negatives = think-final '".' tokens max -36); w2 col = E[<|im_end|>] - E[newline tokens].
## Pass 3 plan (read, edited v2a): work/req_p3.json -- 13 pass-1 natural texts + 10 O2 texts; token_logprobs + topk20 all positions.
Measure: P('"The') at </think>, P(eos) after '".', thinking per-token TV vs pass 1 (expected sampling flips).

## Pass 3 results (read, edited v2a) -> work/fwd_03 (code/023-025)
- At </think>: P('"The') = 1.000 in 13/13 (was e^-9..e^-16). After forced '"The answer is (X)".': P(<|im_end|>) = 1.000 in 10/10
  (was .05-1.0). The forced ' answer is (X)".' tokens keep their original log-probs (position-local edit).
- EOS gate fires in natural answer text only at its final '**The answer is (X)**' (2/13) -> harmless.
- Thinking perturbation (removed neurons + leak): mean |dlogp| ~5e-4; summed sampling TV over generated thinking tokens
  0.06-0.83 per trace (mean .35) => ~72% of traces expected byte-identical up to </think> (HF estimate).
## Pass 4 plan (generate both, edit v2a unchanged): DEV = the 5 closed-but-cut bank traces + 27 random finished bank traces (seed 6).
Expect: +4 (1740, 9274, 8198, 6014), finished traces same answers, ~240 fewer tokens each.
- pass 4 first attempt FAILED in server quantize (disk space, not charged); freed 1.3 GB of my scratch (fwd_02 activations; kept O2/opener content rows in work/p2_L23_content_rows.npy), resubmitted.
## Pass 5 plan (generate both, edit v2a unchanged): HOLDOUT A = fresh random 32 practice questions (sample seed 606). Queued while pass 4 waits.

## Pass 4 results (generate both, v2a, DEV 32 bank) -> work/fwd_04 (code/026)
- acc 71.9 -> 62.5 (-9.4, SE 9.4; +3/-6); cut 15.6 -> 9.4%; tokens -284; 7 finished answers WITHOUT an answer line.
- Mechanism half-worked: every answer starts with '"The' (p0 push OK); '"The answer is (X)"' + end in 9/32 (gains 8198 C, 6014 D:
  both cut-in-answer traces of the parent now finish correct). But after '"The' the model often writes its own explanation in quotes
  ('"The child will have ... codominance."', '"The interest created by ...') and ends -> no answer -> 6 losses (9424, 2817, 6478,
  2602, 2334 + 10935 cut). In generation P(' answer' | '</think>"The') is clearly < 1 (teacher-forced read showed lp -0.1..-7.9).
- Thinking still diverges early in 27/32 (first diff median ~0.2 of the trace) despite HF-estimated TV .35/trace: under Q8 the
  removed neurons' contributions (and Q8 activation-block re-scaling) flip samples much more often than the HF estimate.
- LESSON: queued pass 5 (v2a on HOLDOUT A seed 606) before seeing pass 4 -> pass 5 measures a known-broken edit.

## Edit v3 (code/028, cfg work/edit_cfg3.json) = v2a + third quiet L23 neuron n7602
- gate: PC-cleaned mean-diff of the answer-initial '"The' token (10 samples, never occurs inside thinking where the quote is ' "The')
  vs thinking mean, z*=17, alpha 4 (worst thinking token g ~ -2, 1 of 30k tokens > -5; targets g >= 12 in-sample);
  up = proxy (u 4); w2 col = E[' answer'] - E['"', ' interest', ' child', ...] through the norm gains, Q8-block clipped.
## Pass 6 plan (generate both, v3): HOLDOUT A = same fresh 32 (seed 606) as pass 5 (parent side cached); submitted before seeing pass 5.

## Pass 5 results (generate both, v2a, HOLDOUT A = sample seed 606) -> work/fwd_05
- acc 46.9 -> 46.9 (0.0, SE 4.5; +1/-1); cut 53.1 -> 46.9%; tokens -44. No finished answer without an answer line here.
- Thinking diverges before </think> in 28/32 (often in the first 5-20%), so the paired result is mostly resampling noise.
- Answers start with '"The' everywhere; ' answer' follows only sometimes ('"The answer is (D)"' in 3/17 finished); otherwise a
  quoted explanation that ends with the usual answer line (finished answers still have an answer).
