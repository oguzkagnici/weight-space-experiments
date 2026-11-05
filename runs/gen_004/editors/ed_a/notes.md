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
