# ed_a_g7_short_answer_dupfold_L23: a short visible answer, hosted in freed DUPLICATE neurons of layer 23

TARGETED BEHAVIOUR. After </think> the parent re-derives the whole solution in the visible answer (~250 tokens) before
"The answer is (X)"; in 41 of 43 finished bank traces that letter equals the last letter stated inside the thinking.
When the thinking ends late (>~2,800 tokens) this redundant text is what the 3,072-token cap cuts: 5 of 53 cut bank
traces had closed the thinking, 4 of them with the right answer already stated.

CHANGE (layer 23 MLP only; w1 rows / w3 rows / w2 columns of 3 neurons + one folded w2 column).
1. Free three slots without deleting function: L23 has a cluster of 4 duplicate neurons {1228, 3401, 3793, 6986}
   (cos of w1 rows .9998, w3 rows +-.9999, w2 columns +-.9998; activation correlation .9999). Their contributions are
   algebraically merged into 1228: w2[:,1228] += w2[:,3401] - w2[:,3793] - w2[:,6986] (signs from the duplicate
   structure; no fitting). Q8 simulation: off-target change 2.6e-4 vs 1.0e-3 for removing the 3 quietest neurons.
2. Three gated units, all reading the L23 MLP input:
   - n3401 fires at </think> and writes '"The';
   - n3793 fires at '"The' right after </think> and writes ' answer';
   - n6986 fires at ')".' ending that answer line and writes <|im_end|>.
   Gate (w1 row) = shrinkage-LDA direction (lam .1) target-vs-all-other-tokens on 128k recorded tokens of 61 traces
   minus T x a whitened bias proxy (CV .03), scaled so the gate is -20 at the max negative token and +20 at the
   min positive (2-fold CV by text: pos min z 19-34, neg max z 9-15). silu(-20) quantises to exactly 0 in Q8.
   Up (w3 row) = 4 x bias proxy. Write (w2 column) = gf*(E[target] - mean E[top-10 competitors])/gpf, scaled into
   the Q8 block absmax of every row (block scales preserved exactly; cos .99 to the ideal direction).
   The answer letter is copied from the thinking's last line by the model itself (P = 1.00).

EVIDENCE.
- Read pass (46 synthetic post-think answers): P('"The'|</think>) = P(' answer') = P(<|im_end|>) = 1.000 (min);
  thinking tokens mean |dlogp| 2.7e-4 (fp32 engine), no firing on 30 unseen cut traces (~100k tokens).
- DEV (32 bank questions incl. the 5 closed-but-cut targets): 11 -> 16 correct, +15.6 pts (+6/-1); the 4 targets with
  a stated answer (1740, 6014, 8198, 9274) now finish with '"The answer is (X)"'; tokens -273.
- HOLDOUT (64 fresh practice questions, seeds 777/778): 27 -> 25, -3.1 pts (SE ~3.8; +2/-4). Neither holdout sample
  contained a closed-but-cut parent trace, so all 6 swaps are resampling (traces diverged early: only 10/64 thinking
  traces byte-identical, first difference at median 16-19% of the trace). 32 of 33 edited finished answers are exactly
  '"The answer is (X)"' with the parent's letter (one thinking ended in \boxed{F}; unit did not fire, long answer).
- Expected effect on random questions is small (closed-but-cut traces: 5 of 160 questions seen); the edit does NOT
  achieve off-target silence, so selection will mostly measure resampling noise.