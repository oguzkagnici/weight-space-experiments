# Notes of editor ed_a, generation 9


## Plan (gen 9)
Target: commit at the end of an evaluation sweep through the options, WITH an answer statement.
Unit A (L23 free slot): gate = "eval sweep-end paragraph break", writes "Hence". Unit B (L23 free slot): gate =
paragraph-initial "Hence", writes " answer" (-> "Hence answer is (X)", a form the parent itself uses). Then the
parent's L20 wrap-up and L23 short-answer units finish.
Text detector (code/003): eval sweeps found in 21/47 cut and 19/49 finished bank traces (any sweep incl. option
listing: 28/47 cut). 19 cut traces are pure computation traces (no sweep) and are out of reach of this mechanism.
Parent openers after eval sweep ends: Thus 27, We 18, Given 6, Let's 3, other 7.

## Pass 1 (read, original): work/req1.json
103 texts, 192k tokens: probes = bank trace prefix up to an eval sweep-end break + forced "Hence answer is ("
(last eval sweep end of each of 40 traces, plus first for traces with >=2), and full traces of non-sweep bank traces
(7 longest cut ones dropped). Returns token_logprobs, topk 20, input of L23 w1, all positions.
