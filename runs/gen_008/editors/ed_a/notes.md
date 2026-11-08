# Notes of editor ed_a, generation 8


## Diagnosis (no passes)
- code/002: option sweeps ("Let's examine each option", "Option A:" restarts, >300 chars apart). Finished traces: mean 1.26 sweeps; cut: 2.89.
  Cut traces with >=2 sweeps: 33/53. Finished with >=2: 13/43 (8 correct).
- code/003: crude regex for stated letters in cut traces: 35/53 mention some "answer (X)"; first-stated letter = gold 11, last 13 (noisy regex).

## Pass 1 (read, original) -- plan
- texts: trace_text of 80 bank traces (all 43 finished + 37 cut, incl. all 33 cut with >=2 sweeps), 193,736 tokens.
- return: token_logprobs, topk 20, inputs of L23 and L22 feed_forward.w1 (= ffn_norm output), positions all.
- goal: fit a gate for "paragraph break right before a 2nd/3rd option sweep" at the L23 MLP input; see the next-token dist there.

## Pass 1 (read, original; 80 bank traces, 193,736 tokens) -- results
- "Re-sweep start" gate (break before a 2nd/3rd sweep): grouped-CV AUC 0.89 vs other breaks but only 0.64 vs 1st-sweep starts:
  the residual encodes "a sweep starts", not "a sweep starts AGAIN". Abandoned.
- New target: SWEEP END = paragraph break right after the paragraph holding the last option's evaluation (code/010-012).
  In the bank: 171 sweep ends; next paragraph commits ("Thus/Hence/Therefore ... answer") 17/48 in finished traces vs 20/123 in cut.
  Cut traces with a sweep end: 39/53; finished: 25/43. Cut traces continue with "We need ...", "Thus [no answer]", "Given ...".
  Parent next-token at sweep ends: Thus .33, We .26, Given .08, Option .07, So .03, Let .03, Hence .006, Therefore .007.
  (Opener stats over the bank: paragraphs opened by "Hence" contain an answer statement 79%, "Therefore" 63%, "Thus" 21%.)
- L23 MLP-input shrinkage-LDA sweep-end direction: grouped-CV AUC 0.968; in-sample z: sweep ends 14.1, other breaks -0.4;
  at z>8: recall .85, 77 other breaks (mostly "just evaluated candidates" spots), all-token rate 0.10%; prompt tokens max z 3.3.

## Edit v1 (code/014 args THR=8 K=1 C=1 ALPHA=1.5)
- fold L23 n5105 into its duplicate n331 (w2[:,331] += 1.0089*w2[:,5105]);
- n5105: w1 = K(d/s - (m/s+THR) p) (p = min-variance bias proxy, p.x = 1.00 +- .04), w3 = C p, w2 col = 1.5 * unit(g_post*g_final*(E[Hence] - weighted E[We,Given,Option,Let,But,...])), clipped to 0.98 x Q8 block absmax (norm 1.26).
## Pass 2 (read, edited v1): 140 forced-commit probes at sweep ends (k=1: 64, k=2: 43, k=3: 31), text = prefix + "Hence, the answer is ("
