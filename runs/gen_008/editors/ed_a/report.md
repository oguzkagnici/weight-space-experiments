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
