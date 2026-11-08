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
