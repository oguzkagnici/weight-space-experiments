# ed_a, generation 9: "Hence answer" at the end of an option sweep (L23, two gated units)

Candidate: `ed_a_g9_sweep_end_hence_answer_L23` (L23 MLP only, 14,330 weights). All 6 passes used: 2 read, 4 generate. One pass failed in llama-quantize because the disk was full; I freed 1.3 GB in my work/ folder and resubmitted it.

**Result: the mechanism works as designed, but it does not raise accuracy.**
- Holdout bank (96 questions, measured once): **−2.1 pp (SE 3.9, +6/−8)**.
- Main bank (96 questions): +4.2 (SE 3.6, +8/−4).

## What the model gets wrong
- **It deliberates after evaluating every option.** On knowledge questions the parent goes through the options one by one. At the paragraph break after the last option of such an evaluation sweep, its next token is "Thus" (0.42), "We" (0.24) or "Given" (0.12). "Hence" is almost never chosen. It usually keeps deliberating.
- **Coverage.** 21 of the 47 cut bank traces contain an evaluation sweep (28 if you count sweeps that only list the options). The other 19 cut traces are pure computation (physics, engineering, chemistry, math), and this mechanism cannot reach them.
- **A forced answer at the sweep end** ("Hence answer is (") is right in:
  - 4 of 8 cut traces at the first evaluation sweep end, and 6 of 21 at the last one;
  - 63–67% of finished traces, against 50–68% for the parent.
- **The model hedges even when forced.** After a forced "Hence", the parent writes "," (0.28), " we" (0.18), " the" (0.10) or " answer" (0.03). After "Hence answer" it writes " likely" (0.55) or " is" (0.19). After "Hence answer is" it writes " likely" (0.61), " either" (0.09) or " (" (0.09).

## What I changed
- **Freed two slots.** L23 n5105 and n881 are folded into their duplicate n331: `w2[:,331] += 1.0089·w2[:,5105] + 0.9893·w2[:,881]`. The factors are the w1 ratio times the w3 ratio, taken from the weights.
- **Unit A (n5105): sweep-end break → "Hence".**
  - Gate: a shrinkage-LDA direction at the L23 MLP input that separates evaluation sweep-end breaks from other breaks (grouped-CV AUC 0.991), with threshold z > 8 applied through a minimum-variance bias proxy.
  - Output: "Hence" minus the parent's usual openers.
- **Unit B (n881): "Hence" after a sweep end → " answer".**
  - Gate: AUC 1.0, threshold z > 15. It also fires on the parent's own "Thus" right after a sweep end.
  - Output: " answer" minus ",", " we", " the", " likely".
- **Gate fixes:**
  - Both gate rows are projected off the 4 constant prompt-sink inputs (BOS, `<|im_start|>`, `user`, `\n`).
  - Unit A fired at the prompt's final `<|im_end|>`, where the option list ends. I subtracted 1.5 × that token's mean input direction from its gate.
- **Columns:** each is the unembedding difference multiplied by the post_ffn and final-norm gains, then clipped to the Q8_0 block absmax.
- **Silence:** on 192k recorded tokens the units are nonzero on 0.06% of tokens.

## Evidence

**Read pass (teacher-forced, edited model):**
- P(Hence) at sweep ends rose from 0.000 to 1.000 (median).
- P(" answer") after "Hence" rose from 0.001 to 1.000.

**Generate passes (edited model only, paired against the banks, which the parent reproduces byte-for-byte):**

| set | parent → edited | diff (SE) | +/− | traces with insertion: correct |
|---|---|---|---|---|
| bank ids[:48] (DEV) | 31.2 → 43.8 | +12.5 (4.8) | +6/−0 | 6 → 11 (22 traces) |
| bank ids[48:] | 47.9 → 43.8 | −4.2 (5.1) | +2/−4 | 11 → 9 (27 traces) |
| **holdout 96** | 47.9 → 45.8 | **−2.1 (3.9)** | +6/−8 | 15 → 14 (35 traces) |

- Cut share on the holdout: 42 → 42.
- Byte-identical traces: 12/96 on the holdout and 13/96 on the bank. The first difference comes at a median 16–19% of the trace.

**Why it fails:**
- "Hence answer" is often followed by a hedge ("likely either F or J", "might be (H) or (E)"), and the model keeps deliberating.
- Where it states a single letter, the parent would mostly have reached that letter anyway.
- The rest of the effect is resampling noise. Traces without an insertion went 31 → 30 on the holdout.
- The DEV half's +12.5 was noise as well, which the generation-8 lesson predicted.

## Pitfalls for the next editors
- **Build every edit from `load_original()`, never from `load()`.** `load()` returns your current edit, so rebuilding several times compounds folds. This made my pass 2 off-target numbers useless.
- **Two passes can share one output folder.** After a failed pass, the resubmitted pass got the same number (6) as a pass that was already queued, and its results overwrote that pass's `work/fwd_06` jsonl. Copy results out as soon as a pass finishes.
- **Watch the disk.** llama-quantize needs about 1.9 GB free, and a full disk fails the pass.

## What I would try next
- **Do not push for more commits.** The model is torn between two or three options at these points. Forcing a single letter ("Hence answer is (") converts a hedge into a guess that is right about 30–50% of the time, and it would cost the finished traces that currently resolve the hedge correctly. I estimated the net effect as about 0, so I did not build it.
- **Help the choice instead of stopping it.** The open problem is how good the decision is between the finalists, not when to stop. Mechanisms that improve the choice (generation 3's unused option-text knowledge idea) look more promising than further stop or commit units.
- **Measure stacks on the holdout early.** Every commit-style unit so far (gen 6–9) gives a bank gain and roughly zero on fresh questions.
