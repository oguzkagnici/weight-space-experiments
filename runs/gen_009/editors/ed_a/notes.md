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

## Pass 2 results (read, edited) -- PARTLY INVALID
- BUG: code/015 used load() (= the current edit) as the base, so after repeated rebuilds w2[:,331] carried the fold 5x.
  Off-target numbers of pass 2 (mean |dlogp| 0.28 from the first token on) are therefore meaningless. Fixed: build from
  load_original(); code/023 verifies only w1/w3 rows 5105, 881 and w2 cols 331, 5105, 881 differ, col 331 = one fold.
- Still valid (unit rows/cols were rebuilt each time): at eval sweep-end breaks P(Hence) 0.000 -> 1.000 (median; >0.5 in 47/54);
  after "Hence" P(" answer") 0.001 -> 1.000 (53/54). After "Hence answer": " likely" .55, " is" .19; after "Hence answer is":
  " likely" .61, " either" .09, " (" .09.
- Edit v1 (fixed build) = current edit. Silence read-out: |h|>0.005 on 0.056% / 0.058% of recorded tokens.

## Pass 3 (generate, edited, bank ids[:48]) work/req3.json; Pass 4 = bank ids[48:] (work/req4.json)

## Pass 3 results (generate, edited v1, bank ids[:48], paired vs bank; code/022, 024)
- 31.2 -> 43.8: +12.5 (SE 4.8), +6/-0; cut 24 -> 22; tokens 2329 -> 2249; 6/48 byte-identical; first diff median 17% of trace.
- 22 traces got an inserted "Hence answer": correct 6 -> 11 (cut 9 -> 8). Without insertion: 9 -> 10.
- What follows "Hence answer": often a confident statement ("likely (E).", "is (F).") that then closes the thinking,
  but often a hedge ("likely either F or J", "might be (H) or (E)") followed by more deliberation; 7 traces stay cut.
- Considered a 3rd unit forcing " is (" after "Hence answer" (turn hedges into a letter): rejected. Hedged cases that
  later finish correct (10964, 2654, 7019, 4811, 2731) would be coin flips, and still-cut traces' forced letters are
  right only ~30-50%: expected net ~0. v1 commits when the model is confident and keeps deliberating when torn.
- DECISION: v1 is final. Queue pass 4 (bank ids[48:]) and passes 5-6 (holdout bank, 48+48) with v1.

## Pass 4 (bank ids[48:]) FAILED in llama-quantize (disk full: 1.4 GB free). Freed 1.3 GB in my work/ (pass-1
activation file, old snapshots) and resubmitted as gen_009_ed_a_p06_8b21ea (does not count twice).

## Pass 5 results (generate, edited v1, HOLDOUT ids[:48], paired vs holdout bank)
- 60.4 -> 52.1: -8.3 (SE 5.8), +2/-6; cut 16 -> 19; 6/48 byte-identical; first diff median 18%.
- 17 traces with inserted "Hence answer": 10 -> 8 (+1/-3). Without insertion: 19 -> 17 (+1/-3, pure resampling: 4206, 4235, 10276 now cut).
- The losses with insertion (11253, 2134, 4740) are hedged statements ("likely (A) or (C)", "likely F. But let's examine nuance")
  followed by more deliberation that ends wrong or cut: the insertion re-samples the rest of the trace rather than committing.
