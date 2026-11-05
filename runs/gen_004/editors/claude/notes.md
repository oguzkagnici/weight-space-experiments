# Notes of editor claude, generation 4

## Diagnosis on the trace bank (free)
- 60 of 96 bank traces state an answer inside the thinking ("answer is (X)"); all 43 finished ones do, 17 of 53 cut ones do.
- Finished traces: the answer changes after the first statement in only 2 of 43 (one wrong->right, one wrong->wrong); median 347 tokens after the first statement (incl. ~245 tokens of the post-</think> answer).
- Cut traces with a statement: first stated answer correct in 9 of 17, lost to the cap. Upper bound of "stop re-checking after a commitment": about +8 of 96.
- After a line stating an answer, the next line starts with a reopening word (But/Let/However/Check/Wait) in 31 of 58 cases, a wrap-up word (We/Thus/Hence/Now) in 20.
- Hypothesis H1: a component writes "reopen / double-check" after a committed answer; damping it specifically at commitment points (not at ordinary paragraph ends) saves the post-commit re-verification budget.

## Pass 1 (read, original): 58 commit points + 25 reopen + 25 close control paragraph ends (texts cut at a line end). Last position: residual 0-24, MLP 0-23, o_proj inputs of all attention layers, top-30, attn_last 10/14/18/22.
