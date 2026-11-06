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

## Pass 3 result (read, edited E2), code/017-018
- P(C) at post-conclusion breaks: reconsider(fin) .14 -> .29 (P(R) .65 -> .50), commit .44 -> .70, reconsider(cut, n=8) .11 -> .13; controls flat (.29->.31, .34->.33); top-1 unchanged at controls except 1 of 7.
- n5407 act: rec(fin) -0.88 -> -5.64, rec(cut) -0.44 -> -1.95, ctrl -0.85 -> -1.60 / -0.43 -> -0.22.
- New cancellation: L22 n3686 act 1.56 -> 6.83 at conclusion points, giving back +0.59 of the -1.96 logits from L18. L19 n50 ablated (no output).
- Weak point: in cut traces the conclusion direction projects less (their "conclusions" are more tentative), so the edit moves them little.
## Edit E3 (current): E2 + model.layers.22.feed_forward.w2.weight column 3686 set to 0 (code/020). Tensors: L18 w3 row 5407 (-0.6 d), L19 w2 col 50 = 0, L22 w2 col 3686 = 0.
## Pass 4 (generate both, DEV32 = work/dev32.json: all 15 DEV cut traces with a conclusion, 3 cut without, 14 finished; enriched for the target).

## Pass 4 result (generate both, DEV32, edit E3), code/021
- Accuracy 28.1% -> 40.6%, paired +12.5 pp (SE 7.4), 5 wrong->right / 1 right->wrong. Cut share 56% -> 47% (-9.4, SE 9.4). Mean tokens 2509 -> 2414 (-95, SE 108).
- Of 18 originally-cut traces, 6 now finish (4 correct: 1171 C, 10018 D, 11121 C, 4192 J; 2 wrong: 4811 A, 10935 H). Of 14 originally-finished: accuracy unchanged 64%, 5822 B->J (w->r), 2147 J -> cut (r->w); 6800 and 1643 (both wrong) became cut. Tokens on that subset 1785 -> 1978 (trajectories diverge once a token differs).
- No lexical degeneration: therefore/thus/hence density per 1k tokens unchanged (no repeated words).
- DEV32 is enriched (all 15 cut traces with a conclusion), so the gain overstates the natural-distribution effect.
## Pass 5 (generate both, HOLDOUT 32 = work/split.json holdout, same edit E3, no changes after DEV).
## Pass 6 (generate both, HOLDOUT2 = 32 fresh non-bank practice questions, work/holdout2.json, seed 2026; same edit E3).
- code/023 on pass 4: post-conclusion commit rate 0.16 -> 0.24 (11/68 -> 11/46); reconsideration episodes after a conclusion 57 -> 35; conclusion points per 1k tokens 0.85 -> 0.60; share of trace after first conclusion 0.38 -> 0.32.
- Pass 5 (HOLDOUT) FAILED on 'No space left on device' (disk full machine-wide; not charged). Deleted my regenerable work/r1_neuron_contrib_L*.npy and the failed job's snapshot to free space. Pass 6 (HOLDOUT2) is running; HOLDOUT to be resubmitted after it (its output dir would collide: copy fwd_06 first).

## Pass 6 result (generate both, HOLDOUT2 = 32 fresh non-bank questions, edit E3) -- copied to work/fwd_06_holdout2
- Accuracy 50.0% -> 43.8%, paired -6.25 pp (SE 6.2), 1 w->r / 3 r->w. Cut 43.8% -> 40.6%. Tokens 2245 -> 2149 (-96, SE 79).
- The 3 r->w (5465, 7862, 9590) are traces that originally finished at 2949/2935/3028 tokens (right at the cap) and now get cut: divergence near the cap.
- Of 14 originally cut, 4 now finish: 1 correct (5779), 3 wrong.
- Mechanism replicates: post-conclusion commit rate .23 -> .31 (12/52 -> 16/52); share of trace after first conclusion .32 -> .29.
## Pass 7 (= budget pass 6; HOLDOUT, the 32 bank holdout questions, same edit E3) resubmitted; out dir work/fwd_06 (collides; holdout2 copied first).
