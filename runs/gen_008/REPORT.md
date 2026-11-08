# Generation 8 report

- Created: 2026-11-07 23:21 UTC
- Parent: `gen005_carry_gate_wrapup_L20`
- Editors: claude, ed_a; budget 6 passes each; up to 1 candidates each
- Hidden selection subset: 128 questions (seed `selection-gen-8`)

## Suggested starting points

- **claude**: Stack mechanisms into one candidate large enough to pass (about +6 points or more). Start from the carried stack (generation 7's short visible answer on L23's freed duplicate slots plus the L22 two-finalist commit) and add in-scope mechanisms that convert DIFFERENT cut traces, for example hypothesis churn (generation 5, L23 n74: 'Alternatively/Perhaps' after uncertainty statements) or hedged answer statements re-verified until the cap. Size each mechanism first on the bank (how many cut traces it can convert, how often the converted answer is right, how many finished-correct traces it risks), then measure the whole stack paired on all 96 bank questions with edited-only generate passes before a fresh holdout.
- **ed_a**: Repeated option sweeps in knowledge questions (law 9 of 12 and health 9 of 11 bank questions are cut): cut traces sweep the full option list about 3 times versus 1.2 in finished ones ('Let's examine each option', '(A) ... (B) ...' restarts). Deliberating again over options already evaluated is in scope. First measure what a commit at the start of the 2nd and 3rd sweep would be worth (forced-answer accuracy there; a cut scores 0, so converting a trace is worth its answer's accuracy), then build a gated repair that fires at a re-sweep start and writes a commit to the current best option, and size it on all 96 bank questions with edited-only generate passes (paired against the bank for free). Remember the size target: about +6 points to pass selection.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | edited | 54 | read 54 texts | 950.6 |
| claude | 2 | read | edited | 54 | read 54 texts | 1053.9 |
| claude | 3 | generate | edited | None | read None texts | 1883.3 |
| claude | 4 | generate | edited | None | read None texts | 1719.5 |
| claude | 5 | generate | edited | None | read None texts | 2059.9 |
| claude | 6 | generate | edited | None | read None texts | 2105.6 |
| ed_a | 1 | read | original | 80 | read 80 texts | 1923.6 |
| ed_a | 2 | read | edited | 140 | read 140 texts | 1909.1 |
| ed_a | 3 | generate | edited | None | read None texts | 1957.7 |
| ed_a | 4 | generate | edited | None | read None texts | 2161.4 |
| ed_a | 5 | generate | both | None | read None texts | 3516.4 |
| ed_a | 6 | generate | both | None | read None texts | 2891.4 |

## Candidates

### carry/stack_shortans_L23_finalist_L22

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight, model.layers.22.feed_forward.w1.weight, model.layers.22.feed_forward.w2.weight, model.layers.22.feed_forward.w3.weight
- Weights changed (bf16): 26596

# stack_shortans_L23_finalist_L22: generation 7's two in-scope mechanisms stacked on the parent

The two mechanisms touch disjoint tensors (layer 23 and layer 22), so the stack is a plain union of the two edits.

1. **`ed_a_g7_short_answer_dupfold_L23`** (`ed_a`, generation 7).
   - Slots are freed in a cluster of near-duplicate L23 neurons: `w2[:,1228] += w2[:,3401] - w2[:,3793] - w2[:,6986]`.
   - Three gated units fill them: `</think>` -> '"The', '"The' -> ' answer', and ')".' -> `<|im_end|>`. The visible answer becomes exactly "The answer is (X)", with the letter copied from the thinking.
   - It targets traces that close the thinking late and are then cut inside the ~250-token re-derivation.
   - Generation-7 selection: +3.1 pp (SE 2.9, z 1.08, +9/-5).
2. **The two-finalist commit at L22 n1786** (`claude`, generation 7; built but not submitted).
   - The gate is the mean-difference direction between paragraph breaks that end a two-finalist statement and other breaks (LOTO AUC 0.864), with a bias-proxy threshold of 7.5 and k 6.
   - The gate row is projected off the prompt-sink subspace, so prompt tokens do not fire it.
   - w3 = mu_hat/mp; the w2 column writes commit openers minus deliberation openers, clipped to the Q8 block absmax.
   - Paired on all 96 bank questions: +1.0 pp (SE 2.8, +4/-3). After a two-finalist paragraph, a commit opener follows in 10/29 cases vs 3/39 for the parent.

**Why carry it:** each mechanism converts a different, rare kind of cut trace. Every edit resamples traces, so neither can pass alone. The stack tests whether their gains add up on a fresh selection set.

### claude/claude_g8_stack_commit_opener_L23

- Tensors: model.layers.22.feed_forward.w1.weight, model.layers.22.feed_forward.w2.weight, model.layers.22.feed_forward.w3.weight, model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 36834

Stack: carried gen-8 stack (gen-7 claude L22 two-finalist gate n1786 + gen-7 ed_a L23 short visible answer on freed
duplicate slots) PLUS a new L23 commit-opener unit on n881.

Target behaviour: after a paragraph that narrows the answer to two finalists ("so the answer is either B or D"), the
parent opens the next paragraph with "Thus"/"So" and then continues deliberating (" we", " let", " which option",
" maybe") until the 3,072-token cap cuts the trace (a cut scores 0). The unit makes that opener a decision
("Thus I'll go with (B)" / "Thus answer is (B)"); the parent's L20 wrap-up unit then closes the thinking.

Change (L23 MLP only, on top of the carry):
- Freed n881: it is a near-duplicate of n331/n5105 (h_881 = 1.139 h_331 - 0.158 h_5105, residual 1.4%);
  w2[:,331] += 1.139 w2[:,881], w2[:,5105] += -0.158 w2[:,881].
- n881 gate (w1 row): direction at the L23 MLP input separating "Thus"/"So" after a two-finalist paragraph from 103
  other paragraph-initial openers (LOTO AUC 0.963), bias proxy (mu_hat) with threshold 6.0, prompt-sink subspace
  projected off and penalised; w3 row = mu_hat/|mu|; w2 column (norm 1.43, clipped to Q8_0 block absmax) writes
  " I"/" answer" minus " we"/" let"/" which"/" option"/" what"/" list"/" maybe"/" need".
- Built by code/004_build_opener_unit.py 6.0 0.5 4 2 (fold: code/003).

Evidence (edited-only generate passes on the 96 trace-bank questions, same seeds, paired):
- Teacher-forced (pass 2): at the target openers P(" I"/" answer") 0.096 -> 0.779, P(" we") 0.361 -> 0.029.
  Elsewhere mean |dlogp| 0.00021 per token (99,534 tokens).
- Stack+unit vs parent (bank): all 96 +12.5 pp (SE 4.0, z 3.13), +14/-2, cut 53 -> 43.
  REST48 (no overlap with the fit): +10.4 pp (SE 4.5), +5/-0.
- Unit's own effect, stack+unit vs carry alone (passes 3-6): all 96 +11.5 pp (SE 3.9, z 2.95), +13/-2, cut 47 -> 43;
  REST48 (out of sample) +8.3 pp (SE 4.0, z 2.07), +4/-0; EXTRA16 +12.5 (+2/-0); DEV32 (in-sample: all 22 fit traces)
  +15.6 (+7/-2). 14/96 traces byte-identical to the carry.
- Carry alone vs parent on the same 96: +1.0 pp (SE 3.5), +6/-5 (its gen-8 selection: +5.47, z 2.38).

### ed_a/ed_a_g8_sweep_end_hence_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 8191

# ed_a_g8_sweep_end_hence_L23: commit at the end of an option sweep ("Hence") instead of deliberating again

TARGETED BEHAVIOUR. On knowledge questions (law, health, psychology, economics) the parent evaluates the options one by one
("Option A: ... Option J: ..."). Then, instead of concluding, it opens another round ("We need to determine ...", "Given ...",
"Let's examine each option ...") and runs to the 3,072-token cap. Cut bank traces sweep the options 2.9 times on average;
finished traces sweep 1.3 times. At the paragraph break right after a sweep's LAST option ("sweep end"), the next paragraph
commits ("Thus/Hence/Therefore ... answer") 17 of 48 times in finished traces but only 20 of 123 times in cut traces.
39 of 53 cut bank traces contain a sweep end. Parent next token there: Thus .33, We .26, Given .08, Option .07, Hence .006.
Paragraphs that open with "Hence" state an answer 79% of the time ("Thus": 21%).

VALUE OF COMMITTING THERE (pass 2, forced "Hence, the answer is (" at the sweep end, teacher-forced bank prefixes):
- cut traces (which score 0): correct at the 1st sweep end 14/39 (36%), at the 2nd 14/30 (47%), at the 3rd 8/23;
- finished traces: 18/25 at the 1st sweep end, against 19/25 for the trace's own answer (same letter 21/25).

CHANGE (layer 23 MLP only; 3 tensors, about 8.2k weights).
1. Free a slot: L23 n5105 is a near-duplicate of n331 (cos w1 .9995, w3 .9996, w2 .9995). w2[:,331] += 1.0089*w2[:,5105]
   (the scale is the w1 x w3 projection ratio).
2. Repurpose n5105 as a gated unit:
   - gate (w1 row) = K*(d/s - (m/s + THR)*p), with K=1 and THR=8.
     d is a shrinkage-LDA direction at the L23 MLP input: sweep-end breaks (168) against all other paragraph breaks (2,789),
     from pass-1 activations of 80 bank traces. Grouped (by trace) 5-fold CV AUC is 0.968.
     m and s standardise d over all tokens. p is a minimum-variance bias proxy (p.x = 1.00 +- .04).
     In sample, z > 8 catches 85% of sweep ends, 77 other breaks (mostly "just evaluated the candidates" spots) and 0.10% of
     all tokens. The prompt and sink tokens reach at most z 3.3.
   - up (w3 row) = p, so the unit outputs about silu(z - 8).
   - down (w2 column) = 1.5 * unit(g_post23 * g_final * (E[Hence] - the prob-weighted mean of E[We, Given, Option, Let, But,
     The, Now, Between, Our, Which, Both, I, -])), clipped to 0.98 x the absmax of the other 31 entries of its Q8_0 block
     (norm 1.26).
   An L23 MLP unit only changes the logits at the positions where it fires.

MECHANISM MEASURED (pass 2, read, edited). P(Hence) at the sweep-end break rises with the gate score: z 10-12: .15;
z 12-14: .62; z 14-16: .94; z >= 16: .99. Off-target: mean |dlogp| is 0.0039 per token, coming from the fold and gate leak.
In the generations, 67 of 96 edited bank traces contain an inserted "\n\nHence" paragraph. 74 of 160 such paragraphs state an
answer or a letter; "Hence we need ..." (no commit) is the most common non-committing form.

EVIDENCE (thinking mode, cap 3,072, fixed seeds, paired against the parent).
- All 96 trace-bank questions (edited-only generate, paired against the byte-identical bank; DEV):
  38.5 -> 44.8, **+6.2 (SE 3.2), +8/-2**; cut 53 -> 50.
  - Half A: +10.4 (SE 4.5, +5/-0). Half B: +2.1 (SE 4.7, +3/-2).
- Fresh HOLDOUT (64 practice questions not in the bank, both models generated):
  46.9 -> 43.8, **-3.1 (SE 3.8), +2/-4**.
  - Holdout A: 0.0 (+2/-2). Holdout B: -6.2 (+0/-2).
- Only 4/96 bank and 3/64 holdout traces stay byte-identical; the first difference comes at a median 12-20% of the trace.
- Attribution:
  - Swaps where the edited trace contains inserted Hence paragraphs: +8/-2.
  - Swaps without any: +2/-4. All 4 holdout losses are finished-correct traces resampled to the cap with no Hence inserted.
- Combined over 160 questions: +2.5 points.

## Selection (hidden questions) and decision

Parent accuracy on the 128 selection questions: **41.41%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| carry/stack_shortans_L23_finalist_L22 | 46.88 | +5.47 | 2.3 | +2.38 | 8 | 1 | +0.0000 | 26596 |
| claude/claude_g8_stack_commit_opener_L23 | 46.09 | +4.69 | 3.1 | +1.51 | 11 | 5 | +0.0000 | 36834 |
| ed_a/ed_a_g8_sweep_end_hence_L23 | 44.53 | +3.12 | 2.5 | +1.25 | 7 | 3 | +0.0000 | 8191 |

**Decision:** winner `carry/stack_shortans_L23_finalist_L22` becomes the new champion.

## Editors' reports

### claude

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

### ed_a

# ed_a, generation 8: commit at the end of an option sweep (L23 "Hence" unit)

## What the model gets wrong
- **It deliberates over options it has already evaluated.** On knowledge questions the parent goes through the options one by one ("Option A: ... Option J: ..."). Then it opens another round ("We need to determine ...", "Given ...", "Let's examine each option") and runs into the cap.
  - Cut bank traces sweep the options 2.9 times on average; finished traces sweep 1.3 times.
  - 39 of 53 cut traces contain at least one complete sweep.
- **It rarely commits at the end of a sweep.** At the paragraph break right after a sweep's last option, the next paragraph commits 17 of 48 times in finished traces but 20 of 123 times in cut traces.
  - Parent next-token probabilities there: Thus .33, We .26, Given .08, Option .07, Hence .006.
  - Paragraphs that open with "Hence" state an answer 79% of the time; paragraphs that open with "Thus" do so 21% of the time.
- **Committing there is worth a lot** (pass 2: forced "Hence, the answer is (" at sweep ends of bank traces):
  - In cut traces, which score 0, the forced answer is right 36% of the time at the 1st sweep end, 47% at the 2nd and 35% at the 3rd. That is better than the 19-26% of gen 5's random mid-trace probe.
  - In finished traces it costs about 1 in 25 answers (18/25 against 19/25; same letter 21/25).
- **"Re-sweep start" is not a usable target.** A gate for "a 2nd sweep starts" reaches AUC 0.89 against other breaks but only 0.64 against 1st-sweep starts. The residual encodes "a sweep starts", not "starts again". The sweep END is sharp: grouped-CV AUC 0.968 at the L23 MLP input.

## What I changed (`ed_a_g8_sweep_end_hence_L23`; L23 MLP only, 8,191 weights)
- **Fold a duplicate neuron into its twin.** L23 n5105 is a near-duplicate of n331 (cos about .9995). Change: `w2[:,331] += 1.0089*w2[:,5105]`.
- **Repurpose n5105 as a gated unit:**
  - gate (w1 row) = shrinkage-LDA direction "sweep-end break vs other breaks", with z > 8 applied through a minimum-variance bias proxy;
  - up (w3 row) = that proxy;
  - output (w2 column) = 1.5 × (E[Hence] − the probability-weighted mean of E[We, Given, Option, Let, But, …]), scaled through the post_ffn and final-norm gains and clipped to the Q8_0 block absmax.
- **The gate stays off elsewhere:**
  - it fires on 0.10% of tokens;
  - prompt and sink tokens reach at most z 3.3;
  - its false positives are mostly "just evaluated the candidates" breaks.
- Dose (pass 2): P(Hence) at sweep ends is .15 at z 10-12, .62 at 12-14, .94 at 14-16 and .99 at z ≥ 16.

## Evidence
| set | parent → edited | diff (SE) | +/− | cut |
|---|---|---|---|---|
| bank half A (48, DEV, edited-only vs bank) | 43.8 → 54.2 | +10.4 (4.5) | +5/−0 | 24 → 20 |
| bank half B (48) | 33.3 → 35.4 | +2.1 (4.7) | +3/−2 | 29 → 30 |
| **all 96 bank** | 38.5 → 44.8 | **+6.2 (3.2)** | +8/−2 | 53 → 50 |
| holdout A (32 fresh, both) | 37.5 → 37.5 | 0.0 (6.4) | +2/−2 | 19 → 18 |
| holdout B (32 fresh, both) | 56.2 → 50.0 | −6.2 (4.3) | +0/−2 | 12 → 14 |
| **holdout 64** | 46.9 → 43.8 | **−3.1 (3.8)** | +2/−4 | 31 → 32 |

- **Byte-identical traces:** 4/96 on the bank and 3/64 on the holdout. Traces first differ at a median 12-20% of the trace, the same resampling noise as in generations 6-7 (fold plus leak: mean |dlogp| 0.0039 per token).
- **Attribution:**
  - Swaps where the edited trace contains inserted "Hence" paragraphs: **+8/−2**.
  - Swaps with none: **+2/−4**. All 4 holdout losses are finished-correct traces resampled to the cap with no Hence inserted.
- **What the inserted paragraphs say:** 67 of 96 edited bank traces contain them. Only 74 of 160 state an answer; the most common form is "Hence we need ..." (36 of 160), which keeps deliberating. Both mechanism losses on the bank (10393, 2147) are traces that collected 4-7 "Hence we ..." paragraphs and then hit the cap.
- **Honest summary:** the mechanism looks real (cut traces converted to correct answers where it fires). The measured net effect is +6.2 on DEV and −3.1 on fresh holdout, +2.5 over all 160 questions. Expect about +0 to +3 in selection.

## What I would try next
1. **Make "Hence" commit.** Add a second L23 unit that fires at "Hence" right after a sweep-end break and writes "," / " the" / " likely" against " we". Free it from the same duplicate cluster (n881 is also a duplicate of n331). That would turn many of the ~36 "Hence we need" paragraphs into answer statements, which the parent's L20 wrap-up unit then closes.
2. **Stack with the parent's other commit mechanisms.** This edit covers the never-concluding sweep traces, which earlier edits did not reach. It touches L23 n5105/n331 only, so it can be combined with the L22 two-finalist and L23 short-answer units (n3401/3793/6986).
3. **Raise THR slightly for the 1st sweep end.** The 2nd sweep end is worth more (47% against 36%), but the gate cannot count sweeps. A sweep-count feature could come from an attention head that reads earlier "Option J" tokens.

