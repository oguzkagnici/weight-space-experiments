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
