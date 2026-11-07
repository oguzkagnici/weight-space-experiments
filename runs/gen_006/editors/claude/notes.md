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
