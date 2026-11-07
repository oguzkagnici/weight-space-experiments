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
