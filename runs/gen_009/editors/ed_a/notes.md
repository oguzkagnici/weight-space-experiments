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

## Pass 1 results (read, original; code/009-014)
- Parent next token at eval sweep-end breaks (61): Thus .42, We .24, Given .12, Let .03 ...; Hence ~0.
- After forced "Hence": "," .28, " we" .18, " the" .10, " likely" .09, " answer" .03. After "Hence answer": " likely" .61, " is" .13.
  After "Hence answer is": " likely" .63, " either" .11, " (" .05.
- Forced letter after "Hence answer is (": cut traces argmax right 4/8 at first eval sweep end, 6/21 at last; finished traces 0.63-0.67 (parent .50-.68).
- Gate A (L23 w1 input, shrinkage LDA, eval sweep-end break vs other breaks): grouped-CV AUC 0.991; z>8 fires on 87/192k tokens, 58/61 positives; the rest are sweep ends in other formats.
- Gate B (forced paragraph-initial Hence after sweep end vs all): AUC 1.000; z>15 also fires on natural "Thus" right after sweep ends (good) and some natural para-initial "Hence".
- Pitfalls fixed: bias proxy is low (0.27-0.57) on the 4 constant prompt-sink tokens -> project gate rows off them;
  gate A fired at the prompt-end <|im_end|> (pre up to +1.6) -> subtract 1.5 x mean <|im_end|> input direction, then project.

## Edit v1 (code/015, cfg thrA 8, thrB 15, k 4, c3 16, clip_ratio 3, beta 1.5)
- L23: w2[:,331] += 1.0089 w2[:,5105] + 0.9893 w2[:,881] (fold duplicates).
- n5105 (unit A): w1 = 4(dA - (m0+8)p) - 1.5 v_imend, projected off sinks; w3 = 16p; w2 col = gpf*gfn*(E[Hence] - sum q E[other openers]), clipped to Q8 block absmax (|col| 1.23).
- n881 (unit B): same with dB, thr 15; col -> E[" answer"] - (",", " we", " the", " likely", ...) (|col| 0.83).
- Read-out check: |h|>0.005 on 0.056%/0.058% of recorded tokens; at positives h median 224 / 674.

## Pass 2 (read, edited): work/req2.json = 54 probe texts + 25 finished full texts (same as pass 1).
