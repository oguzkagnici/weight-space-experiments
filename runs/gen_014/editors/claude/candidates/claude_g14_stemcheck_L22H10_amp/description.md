# claude_g14_stemcheck_L22H10_amp: NEGATIVE RESULT for the stem-check line; minimal measured edit
Edit: model.layers.22.self_attn.o_proj.weight, columns 1280:1408 (the 128 inputs of head 10) x1.5. Nothing else.
Target: the STEM CHECK (do wrong samples fail to consult the decisive cue in the question stem while judging options?).
Evidence that the line is closed (all on practice questions, parent):
- Text level (31 finished-wrong verbal questions; 15 have a decisive stem cue): every wrong sample USES the decisive cue, usually many times
  (e.g. 6110 'informed' 18-50x, 1409 'WEAKEST' 20x, 6614 'weight/BMI' 37-62x); within 5 mixed cue questions right samples mention it only ~20-30% more.
  Wrong samples see the cue and draw the wrong inference from it; the other ~half of finished-wrong verbal questions are pure recall with no cue.
- Network (pass 2; 98 judging-position prefixes of 35 samples, 10 mixed verbal questions, attn_last at 11 layers): attention from judging positions
  to the stem is ~1% of the mass and slightly HIGHER in wrong samples (.0113 vs .0096 at the gold option's first evaluation); within-question
  right-wrong differences: no head |z| > 3 of 176; split-half correlation of head differences -0.07 / +0.06 (no replicable pattern).
- CAUSAL INSERTION GATE (pass 1; 32 finished-wrong samples of the 15 cue questions): inserting 'Let me re-read the key part of the question: "<cue>"'
  vs the same frame quoting a non-decisive stem sentence, then 'The answer is (': P(gold) among letters +0.041 (SE 0.039) early (~30% of the trace),
  +0.085 (SE 0.066) late (commit paragraph, 12 samples); vs no insertion -0.001 (SE 0.066). Top-1 becomes gold in +2/32 and +1/12.
  Quoting the cue does not change the conclusion, so no edit that makes the model attend to it can help.
Edit rationale: L22.H10 is the only head whose right>wrong stem attention replicated in both halves of the questions (weak, z 2.75 / 1.31).
Read effect of the edit (pass 3, the 120 pass-1 texts, paired): |dP(gold)| <= .0033 in every condition (SE .0015-.0035), top-1 unchanged: null.
HOLDOUT48 at 8,192 (passes 4-5): overall 62.5 -> 60.4% (-2.08 pts, SE 6.31; wrong->right 4, right->wrong 5);
both-finish (n 38) 73.7 -> 71.1% (-2.63, SE 5.95; wrong->right 2, right->wrong 3); cuts 7 -> 8; 0/48 traces identical, first difference at a median
~2% of the trace: the edit only re-draws the samples (resampling noise). Expect ~0 on selection.
