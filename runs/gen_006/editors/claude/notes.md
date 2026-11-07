# Notes of editor claude, generation 6

Focus: never-concluding knowledge questions (law, health). Parent = gen005_carry_gate_wrapup_L20. Bank = runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl

## Diagnosis on the bank (no passes; code/001-004)
- Law+health: 18 of 23 cut. Cut traces sweep the options ~3 times (finished 1.2); per 1k tokens the gap is small.
- Re-sweep openers ("Let's examine each option ...") after >= 5 options seen: cut 0.21/1k tok (23/53 traces), finished 0.16/1k (8/43; 5 of those correct). NOT a clean discriminator.
- TWO FINALISTS (code/004): 18/53 cut traces narrow to two ("either F or G", "between C and H"), first at 58% of the trace on average, then ~1,300 tokens without deciding; gold in the first pair 11/18. Finished traces rarely narrow (4/43; 2 correct). Candidate failure: oscillating between two finalists until the cap.

## Pass 1 plan (read, original=parent): forced-answer probe at two-finalist paragraph ends
- code/005: 39 texts = prompt + trace to end of paragraph with a two-finalist statement + "\n</think>\n\nThe answer is (" ; plus 18 controls (forced at the last paragraph end of each cut narrowing trace). hidden_states [24] last + topk 20.
- Question: P(gold) within the pair at the narrowing point; does the forced argmax fall in the pair; narrowing point vs end-of-trace.
- Edit: none (empty).

## Pass 1 result (read, parent; code/006) -> work/fwd_01, work/probe1_rows.json
- Forced answer ("\n</think>\n\nThe answer is (" appended) at the paragraph end of the FIRST two-finalist statement:
  - CUT traces (18): argmax correct 8/18; argmax inside the pair 15/18; mean p(pair) 0.84.
    - gold in pair (11): correct 7/11, p(pair) 0.99. gold not in pair (7): 1/7.
  - FINISHED traces that narrow (4): 2/4 correct, the same 2 that finish correctly (7019, 2147).
  - Control, forced at the end of the cut traces: 10/18. So deciding at the narrowing loses little vs. deciding at the cap, and saves ~1,300 tokens.
- => Committing at a two-finalist point could rescue ~8 of the 96 bank questions (currently 0 credit as cut). Includes 3 of the parent's regressions (8262 no: picks F; 7373 yes; 5325 yes).
- Final norm weight mean 1.93 (direct multiplication, not 1+w).

## Pass 2 plan (read, parent): gate localisation
- code/007: 22 narrowing traces + 5 finished + 5 cut others; L21 MLP input (w1 input) at all positions; topk 20 + token logprobs.
- Goal: direction separating the "\n\n" after a two-finalist paragraph from other "\n\n"; what the model plans to write there.
- Edit: none.

## Pass 2 result (read, parent; code/008) -> work/fwd_02, work/gate_dir_L21.npy, work/breaks.json
- Paragraph-break tokens = any token containing "\n\n" (".\n\n", "]\n\n" ... 296 types). 1,599 breaks in the 32 traces; 38 positives (break ending a two-finalist paragraph).
- Mean-difference direction at the L21 MLP input, leave-one-trace-out AUC 0.861.
  - thr = median positive score (4.70): fires on 50% of positives, 3.3% of negatives (52/1561); CV 8.3%.
  - thr = q25 positive (4.06): 74% of positives, 12% of negatives; but in the 5 unrelated finished traces only 1 of 111 breaks, in 5 unrelated cut traces 22 of 254. Most "false" firings are later breaks inside narrowing traces (still deliberating between the two: arguably the same state).
- Planned next tokens at positive breaks: "Option", "(A", "-", "Let", "But", "We" (a new option listing); commit words ("Thus") 0.07-0.83 in a few.

## Off-target check (code/009) and EDIT v1 (code/010_build_finalist_gate.py 4.0 6 2 save)
- Bias proxy: mu_hat = mean L21 MLP-input direction, mean projection mp 2.89 (cv 0.34). Gate score g = d.x - thr*(mu_hat.x)/mp.
  - thr 4.0-4.06: fires on ~66% of positive breaks, ~7% of other breaks, 0.2% of all tokens (mostly ".\n\n"; a few prompt-start tokens where up = mu_hat.x/mp ~ 0, so their output is ~0).
- EDIT v1: L21 neuron 6823 (smallest w2 column norm in L21; no neuron is truly quiet: col norms 0.57-1.5, median 0.88).
  - w1 row = 6 (d - (4.0/mp) mu_hat); w3 row = mu_hat/mp (up ~ 1); w2 col = clip(2 u, +-0.98 Q8 block absmax of the other 31) -> norm 1.40, cos(u) 0.94.
  - u = unit(g_post21 * g_final * (mean E[So, Thus, Therefore, Hence] - mean E[-, We, If, List, But, Let, However, Options, (A, Option, Which, Often, Alternatively])).
## Pass 3 plan (read, edited v1, same 32 texts as pass 2): topk 20 + token logprobs
- Check: P(commit) / P(delib) at the 38 positive breaks vs parent (pass 2); mean |dlogp| on tokens before the first positive break (silence check), and on unrelated traces.

## Pass 3 result (read, edited v1) -> work/fwd_03 (NOTE: positions "last" also limited topk to the last token, so P(commit) at breaks was not recorded; token logprobs cover all positions)
- Positive breaks (38): P(actual next token) 0.56 -> 0.39; where the gate fires the deliberation opener collapses ("Option" .80->.04, .50->.00; "Let" .61->.13; "But" .47->.10); where it does not fire (mid-list "(A", "-") nothing changes.
- Other breaks in narrowing traces: 0.66 -> 0.64. Breaks in unrelated finished traces: 0.822 -> 0.823 (no change). Unrelated cut: 0.58 -> 0.57.
- Silence: mean |dlogp| 0.004 per token before the first positive and in unrelated traces; 0.13-0.19% of tokens change by > 0.1.
## Pass 4 plan (generate both, edited v1 unchanged): DEV32 = the 32 bank questions of passes 2-3 (22 narrowing + 5 finished + 5 cut, work/dev32.json)
- Expect: some of the 18 cut narrowing traces finish (forced-probe says ~8 would be right if they committed at the first narrowing); no change to the 5 finished.
- (orchestrator note) While fixing the disk-full incident I saw the last two lines of ed_a's notes (its pass-4 plan); not used for my edit.

## Pass 4 result (generate both, v1, DEV32) -> work/fwd_04
- acc 7 -> 10 of 32: **+9.4 pp (SE 5.2), +3/-0**; cut 23 -> 19 (-12.5 pp, SE 7.4); tokens -149 (SE 77).
  - Gains (all narrowing traces, cut -> finished correct): 2654 (H), 7691 (B), 8262 (E; one of the parent's regressions).
  - cut -> finished wrong: 5325 (E), 6941 (E). finished wrong -> cut: 1643. The 10 unrelated traces: no grade change.
- All 32 traces diverge early (first difference 18-3105 chars, typically 100-1000): the removed original n6823 output flips sampled tokens (gen-5 lesson), so part of the paired difference is resampling noise.
- Mechanism in the generations (code/013): after two-finalist paragraphs, commit opener next: parent 3/39 -> edited 5/20; answer within 600 chars 2/39 -> 3/20; edited traces narrow less often (11 vs 22 traces).
## Pass 5 plan (generate both, v1 unchanged): HOLDOUT32 = 32 fresh practice questions outside the bank (code/012, seed claude-gen6-holdout)
## Pass 6 plan (generate both, v1 unchanged): HOLDOUT32-B (code/014, seed claude-gen6-holdout-b), queued while pass 5 waits

## Pass 5 result (generate both, v1, HOLDOUT32 fresh) -> work/fwd_05
- acc 13 -> 15: **+6.25 pp, +2/-0**; cut 17 -> 13; tokens 2,196 -> 2,058.
- All 4 changed questions are traces where the PARENT narrowed to two finalists and ran to the cap:
  - 11199 (philosophy, "between B and C" x3) -> finished C, correct; 4754 (history, "B or C") -> finished B, correct;
  - 609 (business, "B or D") -> finished G, wrong; 8256 (math, "either A or J") -> finished A, wrong (gold J).
- Mechanism (code/013): after two-finalist paragraphs, commit opener next: parent 0/14 -> edited 3/10; answer within 600 chars 0/14 -> 3/10.
- (container restart at ~05:20; pass 6 requeued by the orchestrator, not charged twice)

## Pass 6 result (generate both, v1, HOLDOUT32-B fresh) -> work/fwd_06
- 0.0 pp (SE 4.5), +1/-1; cut 9 -> 9; tokens -34. Only 4 parent traces narrow here; the gain (8843, math) and the loss (6651, health: finished correct -> cut; first difference at char 633) are not gate events (resampling).
- Holdouts combined (64 fresh): +3.1 pp (SE ~3.1), +3/-1.
## SUBMITTED: finalist_commit_L21 = edit v1 (code/015). Report in report.md.
