# Generation 9 report

- Created: 2026-11-08 14:31 UTC
- Parent: `gen008_carry_stack_shortans_L23_finalist_L22`
- Editors: claude, ed_a; budget 6 passes each; up to 1 candidates each
- Hidden selection subset: 128 questions (seed `selection-gen-9`)

## Suggested starting points

- **claude**: Answers stated inside the thinking that still end in a cut (13 of 47 cut bank traces, 7 of them correct; the parent's L20 wrap-up unit should close these but does not). Find out why the wrap-up gate stays below threshold there (hedged statements such as 'Thus likely answer is (E)', firm statements followed by 'But we must double-check'), and extend it so that a stated answer closes the thinking, then stack it with a second in-scope mechanism if coverage allows. First count, on a random sample, how many cut traces contain such a statement and how early, and how often the stated answer is right. Final measurement on the holdout bank.
- **ed_a**: Commit at the end of a full sweep through the options, with a complete answer statement. Generation 8 showed the sweep end is sharply detectable (L23 MLP input, AUC 0.968) and that a forced answer there is right in 36-47% of cut traces, but a gate that only wrote 'Hence' produced a stated answer in less than half the cases ('Hence we need ...'). Make the commit state the answer (for example a second unit that fires on the connective right after a sweep end and writes ', the answer is' / ' the' against ' we'), using free L23 duplicate slots (n331/n5105/n881 are free; the parent uses n1228/n3401/n3793/n6986). Check coverage on a random sample of cut traces. Final measurement on the holdout bank.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 39 | read 39 texts | 872.2 |
| claude | 2 | read | original | 39 | read 39 texts | 783.8 |
| claude | 3 | generate | edited | None | read None texts | 1878.1 |
| claude | 4 | generate | edited | None | read None texts | 2088.7 |
| claude | 5 | generate | edited | None | read None texts | 2024.8 |
| claude | 6 | generate | edited | None | read None texts | 2013.8 |
| ed_a | 1 | read | original | 103 | read 103 texts | 2313.4 |
| ed_a | 2 | read | edited | 84 | read 84 texts | 1542.4 |
| ed_a | 3 | generate | edited | None | read None texts | 2153.3 |
| ed_a | 5 | generate | edited | None | read None texts | 2013.3 |
| ed_a | 6 | generate | edited | None | read None texts | 1856.0 |
| ed_a | 6 | generate | edited | None | read None texts | 1754.5 |

## Candidates

### claude/claude_g9_firm_conclusion_close_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 6142

Unit A: close the thinking right after a firm single-letter conclusion (L23, one repurposed neuron).

CHANGED: model.layers.23.feed_forward.{w1,w3,w2} at neuron 5217 only (one of the quietest L23 neurons: mean |out| 0.028). All other entries are identical to the parent.
- Gate row: w1 = K*(d - th*mu/m), with K 2 and threshold 8.
  - d = unit mean difference at the L23 MLP input: tokens that end a single-letter answer statement inside the thinking ("... answer is (E)." / "I'll go with J.") minus all other tokens.
  - mu/m = mean think-token input, so its dot product c is about 1 (a bias proxy).
  - Leave-one-text-out AUC of d is 0.988 (at L20).
- Up row: w3 = mu/m.
- Write column: w2 = E[</think>] - mean E[reopen words: But, However, Wait, Let, We, Alternatively, ...]. It is mapped through the final norm gain and inverted through post_ffn_norm, at scale 1, and each entry is clipped to 0.98 x the absmax of its Q8_0 block.

TARGET BEHAVIOUR: in cut traces (3,072-token cap, scored 0) the model often states a firm answer ("Thus answer should be (I).") and then reopens ("But let's double-check ..."). It re-deliberates until it hits the cap.
- In finished traces, the first firm conclusion equals the final answer in 38 of 39.
- On read-pass recordings the unit fires on 72% of statement ends, raising P(</think>) from a median of 0.000 to 0.999 there. Only 72 of 69k other tokens change any of the tracked logits (</think> and the reopen words) by more than 0.5, and all 72 are conclusion-like. No effect on prompt, two-letter ("B or D?") or post-think tokens.
- It is not a length trigger: it fires only where an answer has just been stated.

EVIDENCE (generate, det1 Q8_0, cap 3,072, edited only, paired against the parent's byte-identical trace banks):
- HOLDOUT BANK (96 ids, never used to fit): parent 46 -> edited 48 correct, +2.1 pp (SE 3.9, z 0.53), +8/-6; cut 42 -> 38.
  - Halves: +0.0 (SE 5.2, +3/-3) and +4.2 (SE 5.9, +5/-3).
  - 11 of the 42 cut holdout traces now close right after a conclusion, and 7 of those 11 are right. Every rescue closes after "Thus/Therefore ... (X)."
  - Losses: 7 finished traces were resampled into new cuts (5 of them right before), and 1 closed on a wrong conclusion.
- Main trace bank (96; the gate direction was fit on texts from bank traces, so this is not fully out of sample): +8.3 pp (SE 3.2, z 2.60), +9/-1, cut 47 -> 41.
- Removing n5217's original output resamples ~92% of traces (median first divergence at 15% of the trace). That resampling noise (about ±6 swaps per 96) is comparable to the mechanism's net effect, so the true gain is probably small (+2 to +4 pp).

### ed_a/ed_a_g9_sweep_end_hence_answer_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 14330

# ed_a_g9_sweep_end_hence_answer_L23: state an answer at the end of an option sweep ("Hence answer ...")

TARGETED BEHAVIOUR. On knowledge questions the parent evaluates the options one by one ("(A) ... - not relevant", ..., "(J) ...").
At the paragraph break after the last option of such an evaluation sweep it opens with Thus .42 / We .24 / Given .12 (Hence ~0)
and usually keeps deliberating; cut bank traces sweep repeatedly. Gen 8 showed that writing "Hence" alone is not a commit
("Hence we need ..."). This edit writes "Hence" at the sweep end and then " answer" after it, a commit form the parent itself
uses ("Hence answer: (C)", "Thus answer is (B)").

CHANGE (layer 23 MLP only; 3 w2 columns, 2 w1 rows, 2 w3 rows).
- Free two slots in the duplicate group {331, 5105, 881} (w1/w3/w2 cos >= .9995): w2[:,331] += 1.0089 w2[:,5105] + 0.9893 w2[:,881]
  (factors = w1-ratio x w3-ratio from the weights).
- n5105 (unit A): gate w1 = 4*(dA - (m0+8)*p) - 1.5*v_imend, projected off the 4 constant prompt-sink inputs; dA = shrinkage-LDA
  direction at the L23 MLP input, eval sweep-end break vs other breaks (61 vs 2,520 from teacher-forced bank traces; grouped-CV
  AUC 0.991), threshold z > 8; p = minimum-variance bias proxy (sd .025 on thinking tokens); v_imend = mean input at the prompt's
  final <|im_end|> (where the option list ends, the gate fired before this fix). w3 = 16 p. w2 column = post_ffn gain x final-norm gain x
  (E[Hence] - probability-weighted E[Thus, We, Given, ...]), clipped to 0.98 x each row's Q8_0 block absmax (|col| 1.23).
- n881 (unit B): same construction; gate = forced paragraph-initial "Hence" after a sweep end vs all tokens (AUC 1.000), z > 15;
  it also fires on the parent's own "Thus" right after a sweep end. Column -> E[" answer"] - (",", " we", " the", " likely", ...) (|col| 0.83).
- Read-out on 192k recorded tokens: units nonzero (|h| > .005) on 0.06% of tokens; prompt silent.

EVIDENCE.
- Read (teacher-forced, edited): at eval sweep-end breaks P(Hence) .000 -> 1.000 (median; > .5 in 47/54); after "Hence" P(" answer")
  .001 -> 1.000 (53/54). After "Hence answer" the model writes " likely" .55 / " is" .19; after "is": " likely" .61, " either" .09.
- Generate (edited-only, paired against the byte-identical parent banks, same seeds):
  - bank ids[:48] (DEV): +12.5 (SE 4.8), +6/-0; bank ids[48:]: -4.2 (SE 5.1), +2/-4; ALL 96 BANK: +4.2 (SE 3.6), +8/-4, cut 47 -> 44.
  - **HOLDOUT BANK (96 untouched questions, measured once): 47.9 -> 45.8, -2.1 (SE 3.9), +6/-8; cut 42 -> 42; tokens 2087 -> 2055.**
    (holdout half 1: -8.3, +2/-6; half 2: +4.2, +4/-2.)
  - "Hence answer" was inserted in 35 of 96 holdout traces (15 -> 14 correct); the other 61 changed only by resampling (31 -> 30).
- Why it does not help: what follows the forced "Hence answer" is often a hedge ("likely either F or J", "might be (H) or (E)")
  and the model keeps deliberating; where it states one letter it was mostly already going to. The edit fires as designed but
  mostly re-samples the rest of the trace (first difference at a median 16% of the trace; 12/96 traces byte-identical).

## Selection (hidden questions) and decision

Parent accuracy on the 128 selection questions: **49.22%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| claude/claude_g9_firm_conclusion_close_L23 | 48.44 | -0.78 | 2.4 | -0.32 | 4 | 5 | +0.0000 | 6142 |
| ed_a/ed_a_g9_sweep_end_hence_answer_L23 | 46.88 | -2.34 | 2.1 | -1.11 | 2 | 5 | +0.0000 | 14330 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

# Report of editor `claude`, generation 9

Candidate: `claude_g9_firm_conclusion_close_L23` (Unit A).
- Holdout bank: **+2.1 pp (SE 3.9, +8/−6, n 96)**.
- Main bank: +8.3 (SE 3.2), partly in sample.

## What the model gets wrong
- **Cut traces dominate the losses.** About 44% of thinking traces of the gen-8 champion hit the 3,072-token cap and score 0.
- **Where the cut-trace budget goes** (code/005):
  - Cut traces average 49 paragraphs, finished ones 28.
  - Cut traces have 2.6 churn openers per trace ("Alternatively", "Maybe", "Wait", ...), finished ones 0.4.
  - After a paragraph that ends uncertain, a churn opener follows with P 0.12, vs 0.04 otherwise.
- **Firm conclusions that get reopened** (code/010):
  - 8 of 47 cut bank traces state a firm single-letter conclusion ("Thus answer should be (I).") and then reopen ("But let's double-check …"). 5 of the 8 conclusions are right.
  - In finished traces the first firm conclusion is the final answer in 38 of 39.
- **The gen-5 L20 wrap-up gate (n5414) misses these points** (code/004).
  - At line-ending single-letter statements it does fire (z +11 to +14 unhedged).
  - Yet the parent still reopens with P 0.45–0.65 at the next newline.
  - Mid-line statements leave the gate low (z −8 to −14).

## What I changed and why
- **Where:** L23 n5217, one of the quietest L23 neurons; parameters are in work/unitA_params.json.
- **Gate:** fires on the token that ends a single-letter answer statement inside the thinking.
  - Direction: mean difference at the L23 MLP input, LOTO AUC 0.988.
  - Absolute threshold 8 against a bias proxy (mean think-token input).
- **Write:** </think> against the reopen words, inverted through post_ffn_norm and the final norm gain. Each entry is clipped to its Q8_0 block absmax.
- **Pre-screen** (linear read-out on pass-2 recordings):
  - Fires on 72% of statement ends; there P(</think>) goes from a median of 0.000 to 0.999.
  - 72 of 69k other tokens move a tracked logit by more than 0.5, all conclusion-like.
  - Prompt, two-letter ("B or D?") and post-think tokens are untouched.

## Evidence
All passes are generate, edited only, paired against the parent's byte-identical banks (det1, cap 3,072).

| set | parent → edited | diff (SE) | +/− | cut |
|---|---|---|---|---|
| bank targets (8 firm-conclusion cut traces) | 0 → 4 | +50 (18.9) | +4/−0 | 8 → 3 |
| main bank 96 (direction fit on bank texts) | 38 → 46 | +8.3 (3.2) | +9/−1 | 47 → 41 |
| **holdout bank 96** | 46 → 48 | **+2.1 (3.9)** | +8/−6 | 42 → 38 |

- **The mechanism works on held-out questions.**
  - 11 of the 42 cut holdout traces now close right after a conclusion, and 7 of those 11 are right.
  - 7 of the 8 gains are such rescues.
- **Resampling costs as much as the mechanism gains.**
  - Removing n5217's tiny original output changes ~92% of traces (median first divergence at 15%).
  - 5 of the 6 losses are finished, correct parent traces that the resampled run drove into the cap.
  - Only 1 loss closes on a wrong conclusion.
- **Bank-to-holdout drop:** +8.3 → +2.1. This repeats gen 8's lesson that bank gains overstate.

## What I would try next
1. **Stack Unit A rather than test it alone.**
   - It converts about a quarter of cut traces into finished ones at 64% accuracy, and its loss channel is generic resampling.
   - Stacked with another closing mechanism, the cost of resampling is paid once.
   - The gen-8 lesson applies: units that fail alone can pass together.
2. **Extend coverage to mid-line statements and two-letter commits.**
   - Unit A fires on 72% of line-end single-letter statements.
   - It does not fire on "so it's B, or maybe D?" type states, which are the other half of reopened conclusions.
3. **Reduce the resampling cost.**
   - Use one of `ed_a`'s algebraically freed duplicate slots in L23. Merging duplicates changes the output by 2.6e-4 instead of about 0.03.
   - Divergence still happens at Q8, but it might come later in the trace.
4. **Churn openers after uncertain paragraphs** (12% vs 4%) are the largest unaddressed token sink in cut traces. A gate on "paragraph ends uncertain" that writes against the churn openers is the next candidate mechanism.

Disk note: pass 6 first failed in `llama-quantize` (disk full). A GGUF build needs ~5.3 GB of scratch; keep at least 6 GB free.

### ed_a

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

