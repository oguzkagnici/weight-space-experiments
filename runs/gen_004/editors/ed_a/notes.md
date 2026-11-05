# Notes of editor ed_a, generation 4


## Setup
- DEV/HOLDOUT split of the 96 trace-bank questions (code/002, seed 4, stratified by cut): work/split.json, DEV 64, HOLDOUT 32. HOLDOUT traces are not inspected individually.
- Note: render_think ends with "<think>" and raw starts with "<think>" again; trace_text concatenates them (kept as given).

## Diagnosis on the trace bank (free)
- code/001 (all 96, aggregate only): cut traces are wordier (4.0 chars/token vs 3.2 finished: prose, not arithmetic), ~2x as many paragraphs (49 vs 25).
- code/002-003 (DEV only): "conclusion paragraphs" = short paragraphs starting Thus/So/Therefore/Hence that name a letter
  (e.g. "Thus D is best."). 103 such points on DEV. The next paragraph is a reconsideration ("But let's double-check...",
  "Let's verify", "Check", "Option J:", "However") in 84/103, a commit ("Hence final answer: The answer is (X)") in 19.
  Finished traces: reconsider 52 / commit 15. Cut traces: 32 / 4.
- KEY: in all 25 finished DEV traces with a conclusion, the first concluded letter == final answer (25/25). Post-conclusion
  re-verification never changed the answer. In cut traces with a conclusion (10 of 35), the first concluded letter is gold
  6/10. Committing at the first conclusion would rescue ~6 of 64 DEV questions with no loss in this sample.
- 25 of 35 cut DEV traces never reach a conclusion paragraph (endless option-by-option deliberation); that failure is not addressed by this target.

## Pass 1 (read, original): work/req_read1.json, items work/read1_items.json
- 104 prefixes ending at the paragraph-break token ('.\n\n' etc.): 80 post-conclusion (61 reconsider, 19 commit), 24 control breaks after non-conclusion paragraphs.
- last position: topk 50, hidden 0..24, mlp 0..23, o_proj inputs of attention layers, attn_last for all attention layers.
- Infra note (coordinator): read passes ~70 tok/s now; pass 1 (195k tokens) will take ~45 min of server time. Keep later read passes short.

## Pass 1 results (read, original), code/006-014
- Next-token mass at breaks: post-conclusion reconsider points P(R)=.63-.65, P(C)=.08 (cut) / .15 (finished); commit points (finished) P(R) .37 P(C) .50. Controls P(R) .32-.52.
  Top next tokens after a conclusion: But .30, Let .10, Thus .10, Option .05, We .05, Now .05, Hence .05, Therefore .03.
- Block DLA on the R-minus-C logit (reconsider minus commit points): biggest is L18 MLP (+1.14), then L21 MLP +.70, L20 MLP +.57, L22 attn +.42.
- Single neuron: **L18 n5407** (w2 col writes therefore/thus/hence: Therefore rank 9, Hence 12, Thus 14 of 128k when act<0).
  Activation: commit(fin) -2.31, reconsider(fin) -0.98, reconsider(cut) -0.45, control -0.28..-0.95. corr(act, logit-odds C vs R) over conclusion points = -0.77.
  DLA on R-C: commit -1.63, rec(fin) -0.73, rec(cut) -0.33. It is the main "commit" writer; it is weak exactly in the cut traces.
- Its downstream suppressors: L19 n50 and L22 n3686 read its output (cos -.25..-.40 with both w1 and w3) and write AGAINST therefore/hence (z -15/-18 and -7/-14); active mainly at commit points (L22 n3686 act 4.4 vs 1.2 rec vs 0.3 ctrl). Self-suppression like the letter leader-suppressors of gen 1-3.
- Gate/up decomposition of n5407 (exact reconstruction from recorded h19 minus MLP18 output): gate open at all breaks (w1.x 1.1-2.2); the up projection sets the activation (commit -1.16, rec fin -0.54, rec cut -0.14, ctrl -0.2..-0.5).
- A "conclusion just stated" direction d at the L18 MLP input (x-hat space; mean concl - mean ctrl breaks): cross-fitted projection commit 6.7-7.0, reconsider 4.8 (fin) / 3.1 (cut), control 1.4 / -0.45. CV AUC concl-vs-ctrl .89 at h18.
- First-order prediction (no downstream reaction) for w3[5407] -= beta*d: beta .3 -> rec(cut) P(C) .08->.25, rec(fin) .15->.48, controls unchanged (.17-.36). beta .6 -> .38/.70.

## Edit E1 (current): model.layers.18.feed_forward.w3.weight row 5407 -= 0.3 * d  (d = work/d_concl_l18.npy, unit norm, 2048 weights). code/015.
## Pass 2 (read, edited E1): 50 shortest read-1 prefixes (work/read2_sel.json), topk 50, mlp 18/19/22 at last position.

## Pass 2 result (read, edited E1 beta .3), code/017-018
- n5407 activation at post-conclusion reconsider (fin) -0.88 -> -3.26 (as predicted), but P(C) only .14 -> .18 (first-order predicted .48); commit points .44->.53; controls unchanged.
- Cancellation: L18 block gives -1.31 logits on R-C at conclusion points, L19 n50 takes back +0.91 (act -0.09 -> +0.78; +2.1 at commit points). L22 n3686 did not react. Unrecorded L20/L21/attn presumably take the rest.
## Edit E2 (current): E1 with beta 0.6 + model.layers.19.feed_forward.w2.weight column 50 set to 0 (code/019, args 0.6 0.0).
## Pass 3 (read, edited E2): same 50 prefixes, topk 50, mlp 18-22.
