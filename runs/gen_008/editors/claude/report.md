# Report of editor `claude`, generation 8: a commit opener after two finalists (L23 n881), stacked on the carry

Candidate: `claude_g8_stack_commit_opener_L23` = the carried stack (gen-7 L22 two-finalist gate + gen-7 ed_a L23 short
answer) plus one new L23 unit. 6/6 passes used (2 read, 4 edited-only generate on the 96 bank questions).

## What the model gets wrong
- Cut traces (a cut scores 0) often reach two finalists ("so the answer is B or D"), then open the next paragraph with
  "Thus"/"So" and keep deliberating. At those openers the parent's next token is " we" (0.36), " let", " which",
  " maybe"; " I"/" answer" only 0.10. Bank: 39 two-finalist paragraphs in 48 traces, but only 3 commit right after.

## What I changed
- L23 n881 is a near-duplicate of n331/n5105 (residual 1.4%). I folded its output into them and reused it as a gated
  unit: gate = direction at the L23 MLP input that separates "Thus"/"So" after a two-finalist paragraph from other
  paragraph openers (LOTO AUC 0.963), threshold 6 via a bias proxy, prompt-sink subspace removed; output = " I"/" answer"
  minus the deliberation tokens. Off-target: mean |dlogp| 0.0002 per token (the smallest of any edit so far).

## Evidence (paired, same seeds, bank questions)
| comparison | questions | diff (SE) | z | +/− | cut |
|---|---|---|---|---|---|
| stack+unit vs parent | all 96 | +12.5 (4.0) | 3.13 | +14/−2 | 53 → 43 |
| stack+unit vs carry alone (the unit's own effect) | all 96 | +11.5 (3.9) | 2.95 | +13/−2 | 47 → 43 |
| same, out of sample (no fit traces) | REST48 | +8.3 (4.0) | 2.07 | +4/−0 | 26 → 24 |
| carry alone vs parent | all 96 | +1.0 (3.5) | 0.30 | +6/−5 | 53 → 47 |

- DEV32 contains all 22 fit traces (in-sample, +15.6 vs carry). 12 of the 13 gains vs carry are traces that were cut
  or wrong and now finish with "Thus answer is (X)" / "Thus I'll go with (X)".
- I did not run a fresh-question holdout: the two passes went to the carry-alone control, which was needed to tell the
  unit's effect from the carry's.

## Caveats
- The carry passed gen-8 selection (+5.5, z 2.4) but is only +1.0 on the 96 bank questions. Selection-size samples
  are noisy; expect the candidate's selection gain to be smaller than the bank numbers.
- ed_a's gen-8 unit (sweep-end "Hence") repurposes L23 n5105, which my fold uses. To stack both, one needs another
  free slot (e.g. the {1228, 3401, 3793, 6986} or {5821, 2400} duplicate groups).

## What I would try next
1. Stack ed_a's sweep-end commit on a different free L23 slot, and make "Hence" state the answer (only 74/160 do).
2. Extend the opener unit to "Therefore"/"Hence"/"Given" openers after a two-finalist paragraph (the gate reaches 75%
   of positives; the rest stay below threshold).
3. Measure every stack on a fresh holdout as well as the bank.
