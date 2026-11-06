# Notes of editor ed_a, generation 5

## Split
work/split.json: trace bank 96 -> DEV 64 (35 cut, acc .375) / HOLDOUT 32 (17 cut, acc .406), seed 5, stratified by cut.
HOLDOUT traces are not used for any analysis.

## Diagnosis on the trace bank (no passes)
- code/001: verbatim re-copying of the option list is NOT cut-specific (char share 5% cut vs 9% finished).
- code/002/003: repeated option-by-option sweeps: 2.4 per cut trace vs 1.2 per finished, but per 1k tokens the
  same (0.48 vs 0.49 opener lines/1k). Re-sweeps are a symptom of length, not a discriminating behaviour.
- code/004: only 13/52 cut traces ever state a parenthesised letter leaning (first leaning = gold 5/13).
- code/006 (DEV, paragraph openers per 1k tokens, finished vs cut):
  "Alternatively(, maybe)" 0.02 vs 0.26 (1 vs 28 occurrences), "Thus we (need)" 0.04 vs 0.28, "But perhaps" 0 vs 0.10,
  "Better approach" 0.02 vs 0.09, "We can" 0.02 vs 0.14, "Given the" 0.09 vs 0.30, "We need" 0.32 vs 0.61,
  "Let's examine" 0.17 vs 0.33. Commit openers the other way: "Thus answer" .75 vs .15, "Thus final" .21 vs .01.
- Qualitative: two stuck styles. (1) knowledge questions (law/health/philosophy/psych): hypothesis churn: a paragraph
  ends "Not sure."/"Hmm." and the next opens "Alternatively, maybe <new hypothesis>", or a new full option sweep.
  (2) physics/engineering: method thrashing ("Better approach", "Another approach", "Alternatively, maybe they intend").
- Key asymmetry under det1: a cut trace scores 0, so an edit that fires ONLY in states that occur almost only in
  traces that will be cut cannot lose; it can only rescue. The hypothesis-churn restart ("Alternatively" after an
  uncertainty statement) is such a state (28 vs 1 in DEV).

## Plan
Pass 1 (read, original): 64 DEV traces (trace_text), positions all, token_logprobs, topk 20, inputs of
model.layers.20.feed_forward.w1 and model.layers.23.feed_forward.w1. ~168k tokens.
Goal: next-token distribution at every paragraph break; a "churn restart" direction at MLP inputs; check it is
silent on finished traces; find a quiet neuron to repurpose.

## Pass 1 (read, original, 64 DEV traces, all positions) -> work/fwd_01 (L20 inputs compacted to break rows + sample)
- code/009: 2,572 paragraph breaks. Next paragraph is a churn opener (Alternatively/Maybe/Perhaps/But perhaps/Another/
  Better approach/Possibly...) at 126 of 1,706 breaks in cut traces (7.4%) vs 12 of 866 in finished (1.4%).
- code/010: the model's own P(churn opener: Alternatively/Maybe/Perhaps/Another/Possibly) at breaks: cut .051 vs
  finished .008 (already .036 vs .004 in the first quarter of the trace). At churn breaks .34; at commit breaks ~0.
  Mean-difference direction (churn vs other breaks) at the MLP input: CV AUC .92 (L20), .91 (L23).
- code/011: at churn breaks the alternatives are Let .13, B(etter) .11, But .08, Given .04, I .035, Thus only .013.
  Paragraph-initial "Hence"/"Therefore" are pure commit moves (finished 14+8, cut 1+1; always followed by
  "answer"/"final answer"), "Thus" is mixed (in cut traces "Thus we need" 20/126).
- code/012: gate from means (churn-break mean minus other-break mean + churn-break mean minus token mean), held-out
  by trace: fires on ~40% of churn breaks, 0.00% of non-break tokens, ~1-2% of finished-trace other breaks.
- code/013: lens of (commit-break minus churn-break) state at L20/L23: Thus/Hence/Therefore z +12..+14, churn
  openers z -7..-9: the commit-vs-churn choice is one readable direction.
- code/014: L23 n74 is quiet (mean|a| .08, max .58, |a|*||g_post*w2col|| <= .03): repurposed.

## EDIT v1 (current, for pass 2)  code/017_build_edit.py 6.0 55 save
Tensors model.layers.23.feed_forward.{w1,w3,w2}, neuron 74 only (row 74 of w1 and w3, column 74 of w2).
- w1[74] = d - (theta/mp) * mu_hat, d = unit(unit(g1)+unit(g2)), g1 = mean x23 at churn breaks - mean at other breaks,
  g2 = churn-break mean - mean over 60k non-break tokens (x23 = recorded input of L23 w1, DEV traces);
  mu_hat = unit mean token input, mp = mean projection 6.47; theta = 12.20 = 55th pct of churn-break scores.
- w3[74] = mu_hat/mp (~1 everywhere) => act ~ silu(s - theta).
- w2[:,74] = 6.0 * unit(g_post23 * g_final * (mean E[Hence,Therefore] - mean E[Alternatively,Maybe,Perhaps,Another,Possibly])).
- In-sample firing (act>0.1): churn breaks 57% (median act 1.38), other breaks cut 11%, finished 2.3%, commit 0%,
  non-break tokens 0.000%; finished traces with >=1 fire 8/29, cut 28/35.
Pass 2 plan: read, edited, same 64 DEV traces, all positions, topk 20 + token_logprobs: compare with pass 1 at breaks
(effect size per activation, P(Hence/Therefore), P(churn)) and at all other tokens (should be unchanged).

## Pass 2 (read, edited v1, same 64 DEV traces) -> work/fwd_02; code/019
- Firing churn breaks in cut traces (n 71, act 1.51): P(Hence/Therefore) .000 -> .389, P(churn openers) .281 -> .055.
  Finished-trace churn breaks firing (n 8): P(H/T) .005 -> .555. Other breaks firing: cut n 167 (.001 -> .239),
  finished n 17 (.002 -> .181).
- Dose-response by activation: act .1-.5: P(H/T) .015; .5-1: .056; 1-2: .294; 2-4: .913 (P(churn) -> 0).
- Non-break positions: mean |dlogp| 0.0004, only 7 of 165,263 positions with |dlogp| > 0.1. (L23 output only affects
  the next-token distribution at its own position.)
- Side effect: SiLU negative tail slightly lowers commits where the gate is just below threshold: commit breaks
  P(Thus) .580 -> .568 (fin), .551 -> .491 (cut); non-firing churn breaks P(churn) .41 -> .48.

## EDIT v2 (current, for pass 3)  code/020_build_edit_sharp.py 6.0 55 save 4
Same as v1 but gate sharpened: w1[74] = 4*(d - (theta/mp) mu_hat), w3[74] = mu_hat/(4 mp) (negative tail /4).
kappa 6.0, theta 12.20 (55th pct). In-sample: fires on 59% of churn breaks (median act 1.61), 11% of other breaks in
cut traces, 2.3% in finished, 0% commit breaks (fin); 8/29 finished and 28/35 cut DEV traces have >= 1 firing.
Pass 3 plan: generate both on work/dev_gen_set.json = 8 finished DEV traces with a firing (5 correct) + 24 of the 28
cut DEV traces with a firing (code/021). Traces without a firing are expected unchanged.

## Pass 3 (generate both, v2, DEV set of 32: 8 finished-with-firing + 24 cut-with-firing) -> work/fwd_03
- acc 15.6 -> 21.9 (+6.25, SE 6.2; +3/-1); cut 75 -> 65.6%; tokens -197 (SE 99).
- BUT code/023: ALL 32 traces diverged, nearly all at mid-sentence word choices early in the trace (char 81..3200),
  not at gated breaks with Hence/Therefore. These are butterfly divergences: the gains/losses are resampling noise,
  not the mechanism (8198 R->W diverged at 6% of the trace on "occurs at vertex of parabola").
- Cause (code/024): llama.cpp quantises w2 to Q8_0 in blocks of 32 along each row. The new column 74 (kappa 6) exceeded
  the absmax of its block peers (cols 64..95) in 1,450 of 2,048 rows (median ratio 1.8, max 14x), so the block scale
  grew and 31 other neurons' weights in every such block were quantised more coarsely -> small logit changes at every
  token -> early sampled-token flips. PyTorch (read pass) does not show this.
- Lesson for later editors: under det1, keep every new weight inside the Q8_0 block absmax of its neighbours (w2
  columns are the dangerous case; w1/w3 rows own their blocks).

## EDIT v3 (current, for pass 4)  code/026_build_v3.py 55 6.0 save
- Neuron L23 n74. Gate w1[74] = 16*(d - (theta/mp) mu_hat) (alpha 16: SiLU tail negligible), theta 12.20 (55th pct).
- w3[74] = S * mu_hat/(16 mp), S = 4.66 (activation gain).
- w2[:,74] = clip(2*u, +-0.98*peer block absmax per row) (614 rows clipped; ||c|| 1.39, c.u 1.29, lens nearly unchanged:
  Hence +28, Therefore +20, Thus +11, Alternatively -7, Maybe -14, Perhaps -14, Another -11.5).
- S chosen so act*c has the same u-component as v2 (kappa 6). Firing stats as v2 (59% churn breaks, 2.4% finished
  other breaks, 0% commit breaks fin, 0.000% non-break tokens).
- code/027: at generated positions, 20 of 154k have act < -0.047 (min -0.97, numeric tokens like "2." with negative
  bias-proxy projection); harmless for the next digit; prompt positions are irrelevant (L23 output only feeds logits).
Pass 4 plan: generate both, same DEV set (work/dev_gen_set.json) to compare with pass 3: divergences should now
happen at gated breaks.

## Pass 4 (generate both, v3, same DEV set of 32) -> work/fwd_04
- acc 15.6 -> 25.0 (+9.4, SE 5.2; +3/-0); cut 75.0 -> 62.5%; tokens -129 (SE 67).
  W->R: 2147 (psych), 1740 (law), 7615 (econ), all cut -> finished correct. Newly finished wrong: 1375 (law, H), 5822 (other, B).
  6829 (finished wrong) -> cut wrong. No finished-correct trace lost (5 of 5 kept their answer).
- Still all 32 traces diverge, mostly at the SAME early mid-sentence points as with v2 (code/023). The common factor
  is the removal of the original L23 n74 (quietest L23 neuron, max contribution .04): under det1 + top-p/top-k,
  a tiny logit change at any token can flip the sampled token. code/028/029: no L23 neuron has a Q8_0 w2 column that
  quantises to zero, and none is quieter than n74, so a repurposed neuron always causes these resampling divergences.
  => paired differences contain resampling noise (like the old engine) in addition to the mechanism.
- Mechanism is visible in the edited generations (code/030): paragraph-initial churn openers 0.92 -> 0.56 per 1k
  tokens (-39%), paragraph-initial Hence/Therefore 0.10 -> 0.87 per 1k.
- v3 chosen for HOLDOUT (no further tuning on DEV).

Pass 5 plan: generate both, v3 (unchanged), HOLDOUT = the 32 held-out trace-bank questions (work/split.json
holdout: 17 cut, 15 finished; never used in any analysis).
Pass 6 plan (submitted while pass 5 runs, edit unchanged v3): generate both on a fresh random sample of 32 practice
questions (sample n 32, seed 505) as a second holdout.

## Pass 5 (generate both, v3, HOLDOUT 32 bank questions) -> work/fwd_05
- acc 40.6 -> 40.6 (0.0, SE 4.5; +1/-1); cut 53.1 -> 56.2%; tokens +114 (SE 63). 29 of 32 traces diverge, 3 identical.
  W->R 7691 (math, cut -> finished B via "Hence answer c = 2"); R->W 10393 (comp sci, finished correct -> cut after a
  divergence at 49% on an ordinary word). 10770 finished wrong -> cut.
- HOLDOUT traces churn much less than DEV (orig 0.31 vs 0.92 churn openers per 1k): its cut traces are mostly
  math/chemistry/engineering/business computations, where the gate rarely applies.
- New failure mode visible: the gate fires but the state is not ready to commit -> "Hence we need to reason
  logically..." / "Hence we need to infer..." (2055, 4192): Hence used as a connective, not a commit.

## Pass 6 (generate both, v3, fresh random 32 practice questions, sample seed 505) -> work/fwd_06
- acc 43.75 -> 53.12 (+9.4, SE 5.2; +3/-0); cut 50 -> 43.8%; tokens -69 (SE 75). 29 of 32 diverge.
  W->R: 4330 (chem; "Not sure.//Hence I'll go with answer G" = mechanism-driven commit, correct), 11365 (eng),
  933 (law). 5518 finished wrong -> cut wrong. Churn openers 0.37 -> 0.22 /1k; Hence/Therefore 0.32 -> 0.49 /1k.

## Summary of v3
DEV (32, selected traces with a firing) +9.4 (SE 5.2, +3/-0); HOLDOUT bank 32: 0.0 (SE 4.5, +1/-1);
HOLDOUT fresh 32: +9.4 (SE 5.2, +3/-0). Two holdouts combined (64): +4.7 pp (SE ~3.4, +4/-1).
Caveat: nearly every edited trace diverges early (repurposing any neuron perturbs all logits slightly), so these
paired differences include resampling noise; mechanism-driven rescues are clearly identifiable in 7691 and 4330.
Candidate: v3 submitted as ed_a_g5_churn_hence_L23.

## End
Candidate ed_a_g5_churn_hence_L23 submitted (code/031). Big scratch deleted (pass-1 L23/L20 inputs, snapshots 02-05;
snapshot 06 = submitted edit kept). report.md: the tool environment refused to write it (subagent restriction on
report files); the report text is in the final reply to the orchestrator.
