# Notes of editor claude, generation 9

# claude, generation 9: notes

Disclosure: as the GA operator I created the holdout bank. I saw its aggregate numbers once (accuracy and cut share), and no per-question rows or traces. I will use it only for the final measurement.

## Bank analysis (code/001, 002; free)
- 13 of 47 cut traces state "answer is (X)" inside the thinking. Most statements are hedged ("Thus likely answer is (D) or H?", "maybe the answer is ..."), at 30-84% of the trace. The first statement is right in 5 of 13.
- Finished traces:
  - 21 with ≥2 statements and an unhedged first statement: it sits at 97% of the trace and always equals the final answer. The L20 wrap-up already closes these.
  - 8 with a hedged first statement: at a median 63% of the trace, ~1,000 tokens before the close. It equals the final answer in 7 of 8 (first right 4, final right 5).
- Size estimate: closing at a hedged single-letter statement could convert up to ~5 of 13 cut traces on the bank and cost ~1 of 8 finished traces. That is about +4 pp if perfectly targeted.
  - Statements naming two letters ("(C) or H?") must be excluded.

## Pass 1 (read, parent): L20 MLP inputs at all positions + top-20 + token logprobs
- Texts: 24 traces with a mid-trace statement (prefix up to the first statement + 600 characters) and 15 full finished traces with a final statement. 69,386 tokens.
- Goal: profile the L20 n5414 gate at hedged vs firm statements, and find a direction for "single-letter hedged statement just completed".
