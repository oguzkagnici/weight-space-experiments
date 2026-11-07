# Generation 6 report

- Created: 2026-11-07 01:02 UTC
- Parent: `gen005_carry_gate_wrapup_L20`
- Editors: claude, ed_a; budget 6 passes each; up to 1 candidates each
- Hidden selection subset: 128 questions (seed `selection-gen-6`)

## Suggested starting points

- **claude**: the never-concluding knowledge questions: law and health traces are cut 18 of 23 times, and 35 of the 53 cut traces never state an answer at all. These are not arithmetic: the model deliberates option by option, recalling rules or facts and weighing them, without converging. Find what differs, inside the reasoning, between finished and never-concluding traces of these subjects (how options are eliminated, whether eliminations are kept, whether the question's key fact is ever pinned down), find the components involved, and test whether an edit helps such traces reach a correct answer (not just stop).
- **ed_a**: answer statements that do not close the thinking: 18 of the 53 cut traces of the parent state 'the answer is (X)' and still reach the cap. In about half the statement comes mid-trace (19-66% of it); most are hedged ('Thus likely answer is (E)', 'maybe the correct answer is C') and some are firm and followed by re-verification ('The answer is (G). But we must double-check ...'). The parent's wrap-up neuron (L20 n5414) was meant to close after a statement. Find out why it does not here (does its gate fire on hedged statements? what overrides it after firm ones?), and check the parent's 4 regressions on the bank (finished and correct before its edit, cut now). Remember the lesson that closing helps only where the stated answer is likely right: measure how often hedged versus firm mid-trace statements are correct before you act on them.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 57 | read 57 texts | 1158.3 |
| claude | 2 | read | original | 32 | read 32 texts | 1058.5 |
| claude | 3 | read | edited | 32 | read 32 texts | 957.3 |
| claude | 4 | generate | both | None | read None texts | 1510.4 |
| claude | 5 | generate | both | None | read None texts | 2484.6 |
| claude | 6 | generate | both | None | read None texts | 2548.7 |
| ed_a | 1 | read | original | 26 | read 26 texts | 639.0 |
| ed_a | 2 | read | original | 60 | read 60 texts | 1318.5 |
| ed_a | 3 | read | edited | 23 | read 23 texts | 510.9 |
| ed_a | 4 | generate | both | None | read None texts | 1016.8 |
| ed_a | 5 | generate | both | None | read None texts | 3027.1 |
| ed_a | 6 | generate | both | None | read None texts | 1167.6 |

## Candidates

### carry/churn_hence_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 6143

CARRIED from generation 5 (ed_a/ed_a_g5_churn_hence_L23; selection there vs the unedited model: +3.12, SE 1.9, z 1.64, +5/-1). Here it is stacked on the parent gen005_carry_gate_wrapup_L20 (which changed only layer 20; this edit changes only layer 23, tensors copied unchanged) to test whether the two edits add up. Original description:

Targets HYPOTHESIS CHURN inside the thinking: at a paragraph break after an uncertainty statement ("Not sure.", "Hmm."), the parent opens yet another speculative hypothesis or method ("Alternatively, maybe ...", "Maybe ...", "Perhaps ...", "Another approach ...") instead of committing. On the 64 DEV trace-bank traces such churn openers follow 7.4% of paragraph breaks in cut traces vs 1.4% in finished ones (126 vs 12; paragraph-initial "Alternatively" 28 vs 1); the model's own P(churn opener) at breaks is .051 in cut vs .008 in finished traces. Paragraph-initial Hence/Therefore are pure commit moves (finished 22, cut 2; always followed by 'answer'/'final answer').
CHANGE (one repurposed neuron, L23 n74, the quietest L23 neuron; 2,048 weights in each of 3 tensors): model.layers.23.feed_forward.w1 row 74 = 16*(d - (12.20/mp)*mu_hat); w3 row 74 = 4.66*mu_hat/(16*mp); w2 column 74 = clip(2*u, +-0.98 * absmax of the other 31 weights of its Q8_0 block in each row). d = unit(unit(g1)+unit(g2)) at the L23 MLP input, g1 = mean at churn breaks - mean at other breaks, g2 = mean at churn breaks - mean over 60k ordinary tokens (recorded on the 64 DEV traces, pass 1); mu_hat = unit mean input (bias proxy, mean projection mp = 6.47); threshold 12.20 = 55th percentile of churn-break scores. u = unit(g_post23 * g_final * (mean E[Hence,Therefore] - mean E[Alternatively,Maybe,Perhaps,Another,Possibly])). The neuron is ~0 except above threshold (fires on 59% of churn breaks, 11% of other breaks in cut traces, 2.4% of breaks in finished traces, 0% of commit breaks and 0.000% of non-break tokens on DEV).
MECHANISM EVIDENCE (teacher-forced read on 64 DEV traces, v1 with the same gate): at firing churn breaks P(Hence/Therefore) .000 -> .39 and P(churn opener) .28 -> .055, graded by activation (act 2-4: P(H/T) .91); elsewhere mean |dlogp| 0.0004 (7 of 165k positions > 0.1). In generations, paragraph-initial churn openers fall 0.92 -> 0.56 per 1k tokens (DEV) and Hence/Therefore openers rise 0.10 -> 0.87.
MEASURED (generate both, paired, det1): DEV 32 (traces with a firing; used for choosing) +9.4 (SE 5.2, +3/-0); HOLDOUT 32 trace-bank questions 0.0 (SE 4.5, +1/-1); HOLDOUT 32 fresh random practice questions +9.4 (SE 5.2, +3/-0). Two holdouts combined (64): +4.7 points (SE ~3.4, +4/-1), cut share -1.6 points.
CAVEATS: replacing any neuron perturbs every logit slightly, so under det1 nearly all edited traces diverge early on ordinary words (no L23 neuron is dead or quantises to zero); paired differences therefore include resampling noise. A w2 column larger than its Q8_0 block peers made this much worse in an earlier version (block scale change); v3 keeps it under the block absmax. Some firings produce "Hence we need to ..." (connective, not a commit). The gate acts mainly in knowledge questions (law, health, psychology, philosophy, economics); computation-heavy cut traces rarely trigger it.

### claude/finalist_commit_L21

- Tensors: model.layers.21.feed_forward.w1.weight, model.layers.21.feed_forward.w2.weight, model.layers.21.feed_forward.w3.weight
- Weights changed (bf16): 6136

TARGET: oscillating between two finalists until the cap. In the parent's trace bank, 18 of 53 cut traces narrow the choice to two options ('either F or G', 'between C and H'), on average 58% into the trace, then spend ~1,300 tokens re-sweeping the options without deciding; gold is in the first pair in 11/18. Finished traces rarely narrow (4/43). A forced answer at the first two-finalist paragraph end is correct in 8/18 of those cut traces (7/11 when gold is in the pair; p(pair) 0.99), and 10/18 at the cap: deciding there gives up little, while being cut scores 0. CHANGE (one repurposed neuron, L21 n6823, the smallest w2 column in L21; 6,144 weights): w1 row = 6*(d - (4.0/2.89)*mu_hat), d = unit mean-difference direction at the L21 MLP input between paragraph-break tokens that end a two-finalist paragraph (38) and all other breaks (1,561) on 32 bank traces (leave-one-trace-out AUC 0.86); mu_hat = mean input direction (mean projection 2.89, a bias proxy); w3 row = mu_hat/2.89 (up ~1); w2 column = clip(2u, +-0.98 x the Q8_0 block absmax of the other 31 entries per row) (norm 1.40), u = unit(g_post21 * g_final * (mean E[So,Thus,Therefore,Hence] - mean E[-,We,If,List,But,Let,However,Options,(A,Option,Which,Often,Alternatively])). READ (teacher-forced, 32 bank traces): where it fires, the deliberation opener collapses (e.g. 'Option' .80->.04, 'Let' .61->.13); breaks in unrelated finished traces unchanged (.822->.823); mean |dlogp| 0.004 per token elsewhere. GENERATE (paired, det1): DEV32 (bank: 22 narrowing + 10 other) +9.4 pp (SE 5.2, +3/-0), cut 23->19; HOLDOUT32 fresh +6.25 (+2/-0), cut 17->13; HOLDOUT32-B fresh 0.0 (+1/-1; only 4 narrowing traces there). Holdouts combined (64 fresh): +3.1 pp (SE ~3.1, +3/-1). All 5 DEV/HOLDOUT32 gains are traces that narrowed to two finalists in the parent and ran to the cap (the HOLDOUT32-B gain and loss are not: resampling after early divergence); after two-finalist paragraphs the edited model opens with a commit word 11/37 vs the parent's 5/59. As with every repurposed neuron, edited traces diverge early (resampling), which adds noise to all paired numbers.

### ed_a/ed_a_g6_short_answer_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 18427

# ed_a_g6_short_answer_L23: answer = the think's own final line, so the visible answer stops costing ~250 tokens

TARGETED BEHAVIOUR. After </think>, the parent re-derives the whole solution in the visible answer (finished traces: 248 tokens on
average, 61-405) and only then writes "The answer is (X)". In 42 of 43 finished bank traces this answer equals the last answer
stated inside the thinking (the remaining one gave no answer), and almost every thinking ends with the line
'Hence final output: "The answer is (X)".'. When the thinking ends late this re-derivation runs into the 3,072-token cap:
5 of the 53 cut bank traces had CLOSED the thinking (think 2,780-3,050 tokens) and were cut inside the answer; 4 of those 5
had the correct answer settled in the thinking (mmlupro_1740, 9274, 8198, 6014). The thinking itself is not changed or shortened.

CHANGE. Three repurposed quiet MLP neurons of the LAST layer (L23: n4526, n810, n7602; w1 row, w3 row, w2 column each, 6,143
weights per tensor). L23 MLP edits act only on the logits of the position where they fire.
- gates: mean-difference directions of the L23 MLP input (top-200 thinking PCs projected out; recorded activations), with a
  threshold through a PC-cleaned mean-direction bias proxy, z* = 15/22/17 thinking-sd; on 30k recorded prompt+thinking tokens the
  pre-activation is <= -13 / -32 / -2 (one token above -5), on answer positions >= +71 / +124 / +12.
  n4526 fires at the </think> position; n7602 on the answer-initial token '"The' (never occurs inside thinking, where the quote is
  ' "The'); n810 six tokens after </think>, right after '"The answer is (X)".' (hard negatives: the think's own final
  '"The answer is (X)".' lines, max -36, so </think> is never replaced).
- up rows: the bias proxy (constant u ~ 4). w2 columns: (E[target] - weighted E[competitors]) x final-norm gain x post_ffn_norm
  gain, targets '"The', ' answer', <|im_end|>; every entry clipped to 0.98 x the absmax of the other 31 neurons of its Q8_0
  block (no block scale changes).
- Net effect: the answer becomes '"The answer is (X)"' + end, X copied from the thinking (P(letter)=1.00 teacher-forced).
No training or fitting; directions are means/differences/PC projections of recorded activations; thresholds are z-scores.

EVIDENCE (thinking mode, cap 3,072, paired with the parent, same seeds).
- Read (pass 3, two-neuron version): P('"The' at </think>) e^-9..e^-16 -> 1.000 (13/13); P(<|im_end|> after the line) .05-1.0
  -> 1.000 (10/10); thinking token log-probs change by ~5e-4 on average.
- HOLDOUT A (32 fresh practice questions, sample seed 606, never used for design): 46.9 -> 53.1 (+6.25, SE 4.3; +2/-0);
  cut 53.1% -> 40.6%; tokens -265 (SE 103). All 19 finished answers are exactly '"The answer is (X)"'. Gains 7802 (math) and 884
  (law, finishes at 2,951 tokens).
- The two-neuron predecessor (no ' answer' neuron) failed on DEV (32 bank questions: -9.4, SE 9.4, +3/-6): after '"The' the
  model often wrote a quoted explanation and stopped without an answer; that is what n7602 fixes (on the same HOLDOUT the
  predecessor scored 0.0, +1/-1).
CAVEAT. Removing three neurons' small original outputs still changes sampled tokens under Q8: 29 of 32 edited traces diverge
from the parent inside the thinking (median ~10-20% of the trace), so paired differences contain resampling noise.

## Selection (hidden questions) and decision

Parent accuracy on the 128 selection questions: **53.91%**. Rule: z >= 2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| ed_a/ed_a_g6_short_answer_L23 | 54.69 | +0.78 | 3.2 | +0.24 | 9 | 8 | +0.0000 | 18427 |
| claude/finalist_commit_L21 | 53.91 | +0.00 | 4.0 | +0.00 | 13 | 13 | +0.0000 | 6136 |
| carry/churn_hence_L23 | 53.12 | -0.78 | 2.6 | -0.30 | 5 | 6 | +0.0000 | 6143 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

# Report of editor `claude`, generation 6

## What the model gets wrong
- **Law and health questions are cut at the cap 18 of 23 times.** These traces do not do a categorically different thing from finished ones: they sweep the options ~3 times instead of ~1.2, but per token the sweeping and "Let's examine each option ..." restarts are only slightly denser (0.21 vs 0.16 per 1k tokens). Mostly the model simply does not know the fact and keeps deliberating.
- **A specific, fixable failure inside that group: oscillating between two finalists.** 18 of the parent's 53 cut bank traces narrow the choice to two options ("either F or G", "between C and H"), on average 58% into the trace, then spend ~1,300 tokens re-sweeping without deciding. Finished traces rarely narrow (4 of 43): they go straight to an answer.
- **Deciding there would usually beat being cut** (pass 1, forced answer after the first two-finalist paragraph):
  - correct in 8 of 18 cut traces; 7 of 11 when gold is one of the two (probability on the pair 0.99);
  - forcing at the cap instead gives 10 of 18, so deciding early gives up little and saves ~1,300 tokens;
  - the 4 finished traces that narrow keep their 2 correct answers.

## What I changed and why
- **One repurposed neuron, L21 n6823** (`finalist_commit_L21`):
  - gate (w1 row): a two-finalist direction at the L21 MLP input. It is the mean difference between paragraph-break tokens ending a two-finalist paragraph and all other breaks (leave-one-trace-out AUC 0.86). The threshold uses the mean-input direction as a bias proxy;
  - up (w3 row): the bias proxy (about 1);
  - output (w2 column): commit openers (So, Thus, Therefore, Hence) minus deliberation openers (Option, (A, -, Let, But, We, ...). It is clipped to 0.98 × the Q8_0 block absmax of the other 31 entries per row.
- Paragraph-break tokens are any token containing "\n\n" (".\n\n", ":\n\n" ... 296 token types), not only token 8.

## Evidence
| set | parent → edited | diff | +/− | cut |
|---|---|---|---|---|
| DEV32 (bank: 22 narrowing + 10 other) | 7 → 10 | +9.4 (SE 5.2) | +3/−0 | 23 → 19 |
| HOLDOUT32 (fresh) | 13 → 15 | +6.25 | +2/−0 | 17 → 13 |
| HOLDOUT32-B (fresh) | – | 0.0 (SE 4.5) | +1/−1 | 9 → 9 |
| holdouts combined (64 fresh) | | **+3.1 (SE ≈ 3.1)** | +3/−1 | |

- **Read pass (teacher-forced):** where the gate fires, the deliberation opener collapses ("Option" 0.80 → 0.04, "Let" 0.61 → 0.13). Mid-list breaks are untouched, as are breaks in unrelated finished traces (0.822 → 0.823). Elsewhere the mean |Δlogp| is 0.004 per token.
- **In the generations:** after two-finalist paragraphs the edited model opens with a commit word 11 of 37 times, against 5 of 59 for the parent.
  - All 5 DEV and HOLDOUT32 gains are traces where the parent narrowed to two finalists and ran to the cap; they now finish with the right finalist.
  - The HOLDOUT32-B gain and loss do not involve the gate. The traces diverged early (resampling) and took different paths.
- **The effect depends on how many cut-and-narrowing questions a set contains:** 18 of 96 bank questions, 4 or 5 of 32 in the fresh sets. Expect a gain of about +2 to +4 pp on random questions, which is hard to resolve with 128.

## What I would try next
1. **Fire on more of the narrowing breaks.** The gate catches about 66% of two-finalist breaks at this threshold; a direction fit on more traces, or a second layer, would raise that. Hedged single-answer statements ("Thus likely answer is (E)") are a related state worth adding.
2. **Make the commit carry through.** After "Thus" the model sometimes writes "Thus we need to decide between C and H". A second gated neuron, firing on "Thus/So" right after a two-finalist paragraph and writing "the answer is" tokens, would complete the decision.
3. **Reduce resampling noise.** Every repurposed neuron diverges traces early. Use a neuron with near-zero original output, if one can be found with a cheap `mlp` read pass at `positions: "mean"`, or test with larger sets.
4. **The never-concluding knowledge questions without two finalists** (most law/health cuts) still have no lever: the model lacks the fact. Commit-type edits will not rescue them.

### ed_a

# ed_a, generation 6: the visible answer re-derives everything and runs into the cap

## What the model gets wrong
- After `</think>`, the parent writes a full step-by-step re-derivation (about 250 tokens; 61-405) and only then "The answer is (X)".
  - In 42 of 43 finished bank traces this answer equals the last answer stated inside the thinking. The one exception gave no answer.
  - Almost every thinking already ends with `Hence final output: "The answer is (X)".`
- When the thinking ends late, this redundant answer is what hits the 3,072-token cap.
  - 5 of the 53 cut bank traces had closed the thinking (thinking 2,780-3,050 tokens) and were cut inside the answer.
  - 4 of those 5 had the right answer settled in the thinking (1740, 9274, 8198, 6014).
- The suggested starting point (mid-trace answer statements) is weak by comparison.
  - Of the 11 cut traces with a first statement before 75% of the trace, only 4 are correct.
  - Late statements (after 85% of the trace) are correct 8 of 11 times, but most of those also need a short answer to finish in time.

## What I changed (candidate `ed_a_g6_short_answer_L23`)
Three repurposed quiet neurons of the last MLP (L23: n4526, n810, n7602). An L23 edit acts only at the position where it fires.

| Neuron | Fires at | Writes |
|---|---|---|
| n4526 | the `</think>` position | `"The` |
| n7602 | the answer-initial `"The` token | ` answer` |
| n810 | right after `"The answer is (X)".` | `<|im_end|>` |

- Notes on the gates:
  - The answer-initial `"The` never occurs inside the thinking, where the quote tokenises as ` "The`.
  - n810's hard negatives are the thinking's own final answer lines (gate value at most -36 there), so `</think>` is never replaced.
- Each gate is a mean-difference direction of the L23 MLP input with the top-200 thinking principal components projected out.
  - The threshold is a z-score, applied through a PC-cleaned mean-direction bias proxy (coefficient of variation .095).
  - The up row is that proxy.
- Each w2 column is (E[target] − E[competitors]) times the final-norm and post_ffn_norm gains.
  - Every entry is clipped to 0.98 × the Q8_0 block absmax of the other 31 neurons, so no block scale changes.
- The answer becomes `"The answer is (X)"` and ends. The letter is copied from the thinking with P = 1.00.

## Evidence (6 passes)
| Pass | Kind | Result |
|---|---|---|
| 1 | read, original, 13 traces + forced answers | P(" answer" after `</think>The`) = e^-6..e^-19<br>P(end after a forced answer line) .01-.03<br>no Q8-silent L23 neuron or input dimension exists |
| 2 | read, original, 6 openers | `"The` (one token) continues to ` answer is (X)".` by copying the thinking's last line<br>P(end) after it .05-1.0<br>the other openers are far less natural |
| 3 | read, edited (2 neurons) | P(`"The`) 1.000 (13/13), P(end) 1.000 (10/10)<br>thinking log-probs change by ~5e-4 |
| 4 | generate, DEV 32 bank, 2 neurons | −9.4 (SE 9.4; +3/−6)<br>after `"The` the model often wrote a quoted explanation and stopped without an answer (7 no-answer) |
| 5 | generate, HOLDOUT A (seed 606), 2 neurons | 0.0 (SE 4.5; +1/−1) |
| 6 | generate, HOLDOUT A, **3 neurons (submitted)** | **46.9 → 53.1, +6.25 (SE 4.3; +2/−0)**<br>cut 53 → 41%, tokens −265<br>all 19 finished answers are exactly `"The answer is (X)"` |

## Caveats
- **Divergence noise remains.** Even quiet repurposed neurons flip sampled tokens under Q8. 27-29 of 32 edited traces diverge inside the thinking (median about 10-20% of the trace), although the HF-based expected-flip estimate was 0.35 per trace. Paired tests therefore still carry resampling noise.
- **Grader note.** The grader code does not zero answers cut at the cap if the answer section already contains "answer is (X)". I did not rely on this.
- **Possible scope question.** The edit removes the explanation from the visible answer. The thinking is untouched.

## What I would try next
- Stack this candidate with the parent's wrap-up neuron and gen-5's churn neuron: a short answer lets late commits finish.
- Find a way to add a gated function without removing a neuron's original output, which is the source of the divergence noise. Gen-5 and this generation showed no dead or Q8-silent neurons in L23.
- Measure on more questions. Expected gain is about +4 points (about 5% of questions close the thinking but are cut in the answer, roughly 80% of them correct).
- Don't queue a holdout before the DEV result is in. My pass 5 measured a design that pass 4 had already shown to be broken.

