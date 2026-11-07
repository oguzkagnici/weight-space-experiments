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
