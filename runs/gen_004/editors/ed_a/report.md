# ed_a, generation 4 (thinking mode): the conclusion-gated "therefore" neuron and its self-suppressors

## Summary
- **Behaviour targeted:** re-verifying after a conclusion has been stated. After a short paragraph like "Thus D is best.", the parent usually opens a new check ("But let's double-check…", "Let's verify", "Option J: …") instead of committing ("Hence final answer: The answer is (D)").
- **Mechanism found:**
  - One neuron, **L18 n5407**, writes therefore/thus/hence and carries most of the commit drive.
  - Two downstream neurons, **L19 n50** and **L22 n3686**, read its output and write against therefore/hence. They cancel most of any boost to it.
- **Edit E3 (2,048 weights in each of 3 tensors):**
  - Wire a "conclusion just stated" direction into n5407's up projection.
  - Ablate its two suppressors.
- **Results:**
  - The targeted behaviour changes in all three generate sets: the commit rate after a conclusion rose by +.06 to +.08.
  - Accuracy: DEV +12.5 pp (SE 7.4); HOLDOUT +6.25 (SE 4.3); HOLDOUT2 −6.25 (SE 6.2). The two holdouts combined give **0.0 pp (SE 3.9)** over 64 questions.
- **No candidate submitted.** The combined holdout shows no accuracy gain.

## What the model gets wrong (trace bank; DEV = 64 of the 96 questions, the other 32 kept as HOLDOUT)
- Cut traces are prose-heavy: 4.0 characters per token against 3.2 in finished traces, and twice as many paragraphs.
- **Conclusion paragraphs** are short paragraphs that start with Thus/So/Therefore/Hence and name a letter. DEV has 103 of them.
  - What follows: a new reconsideration in 84 cases, a commit in 19.
  - In finished traces: 52 reconsiderations, 15 commits.
  - In cut traces: 32 reconsiderations, 4 commits.
- **Re-verification never changed an answer.**
  - In 25 of 25 finished DEV traces that reached a conclusion, the first concluded letter was the final answer.
  - In the 10 cut traces with a conclusion, the first concluded letter was gold 6 times.
  - Cut traces spend about 1,400 tokens after their first conclusion; finished traces spend about 560.
- **Not addressed:** 25 of 35 cut DEV traces never reach a conclusion. They deliberate option by option in prose, especially in health, law and physics.

## Mechanism (pass 1 read on 104 paragraph-break prefixes; passes 2–3 read with the edit)
**Next token after a conclusion.**
- After a conclusion, the next token is "But" (.30), "Let" (.10), "Thus" (.10), "Option", "Now", "Hence"…
- Probability of a commit opener (Hence/Therefore/Thus/So/Final):
  - .08 at reconsideration points in cut traces;
  - .15 at reconsideration points in finished traces;
  - .50 at commit points.

**L18 n5407, the commit writer.**
- Direct logit attribution on the reconsider-minus-commit logit: the largest block is the L18 MLP (+1.14), and almost all of it is n5407.
- Its w2 column ranks Therefore, Hence and Thus 9th, 12th and 14th of 128k tokens.
- Activation: −2.3 at commit points, −0.98 and −0.45 at reconsideration points (finished, cut), −0.3 to −0.95 at control breaks.
- Correlation with the commit log-odds: −0.77.
- Its gate is open at every break; the up projection sets the activation.

**Self-suppression.**
- L19 n50 and L22 n3686 read n5407's output (cosine −.25 to −.40 with both w1 and w3) and write against therefore/hence (z −15 to −18).
- They fire mainly at commit points (n3686 activation 4.4, against 0.3 at controls).

**"Conclusion just stated" direction d.**
- d is the mean L18 MLP-input x̂ at post-conclusion breaks minus the mean at control breaks.
- Cross-fitted projections: commit 6.7, reconsideration 4.8 (finished) and 3.1 (cut), controls −0.5 to 1.4. AUC .89.

**Causal reads.**

| Edit | Commit probability at reconsideration points (finished traces) | Commit points | Controls | What cancelled the boost |
|---|---|---|---|---|
| E1: w3[5407] −= 0.3·d | .14 → .18 (first-order prediction .48) | — | — | L19 n50 took back +0.91 of the −1.31 logits |
| E2: β = 0.6, plus L19 n50 ablated | .14 → .29 | .44 → .70 | flat | L22 n3686 took back +0.59 |
| E3: E2 plus L22 n3686 ablated | not read; generated directly | — | — | — |

- In cut traces the effect was small (n = 8, .11 → .13), because their conclusions project weakly on d.

## Edit E3 (not submitted)
- `model.layers.18.feed_forward.w3.weight` row 5407 −= 0.6·d. d is unit-norm and saved in work/d_concl_l18.npy (code/013).
- `model.layers.19.feed_forward.w2.weight` column 50 set to 0.
- `model.layers.22.feed_forward.w2.weight` column 3686 set to 0.
- Code: code/020. The edit is still in edited/.

## Evidence (generate "both", paired, same seeds)

| Set | n | Accuracy change (pp) | wrong→right / right→wrong | Cut share (pp) | Tokens | Commit rate after a conclusion |
|---|---|---|---|---|---|---|
| DEV32 (enriched: all 15 DEV cut traces with a conclusion) | 32 | **+12.5 (SE 7.4)** | 5 / 1 | −9.4 | −95 | .16 → .24 |
| HOLDOUT (bank, never inspected) | 32 | **+6.25 (SE 4.3)** | 2 / 0 | −9.4 | −90 | .23 → .29 |
| HOLDOUT2 (32 fresh non-bank questions) | 32 | **−6.25 (SE 6.2)** | 1 / 3 | −3.1 | −96 | .23 → .31 |
| Both holdouts combined | 64 | **0.0 (SE 3.9)** | 3 / 3 | | | |

- **Order of work.**
  - E1 and E2 were measured only with read passes.
  - E3 went straight to DEV generation, with no further tuning after DEV.
  - HOLDOUT2 ran before HOLDOUT, because the first HOLDOUT submission failed on a full disk (not charged).
- **What the edit does in generation.**
  - Cut traces now finish more often: 6 of 18 on DEV, 4 of 14 on HOLDOUT2.
  - Re-checks after a conclusion become rarer: 57 → 35 on DEV.
  - There is no lexical degeneration: therefore/thus density per 1k tokens is unchanged, and no word repeats.
- **Why the gain does not hold.**
  - The cut traces that are rescued are only about 50–60% correct.
  - Sampling diverges once a token differs, so some traces that originally finished just under the cap now hit it. All 3 right→wrong cases on HOLDOUT2 had originally finished at 2,935–3,028 tokens.
  - The net is about zero.

## What I would try next
1. **Target the larger failure:** the 70% of cut traces that never conclude. The same circuit, n5407 and its suppressors, might be gated on a "long deliberation without a conclusion" feature.
2. **Strengthen the gate in cut traces.** Their conclusions project weakly on d, so a direction estimated from cut-trace conclusions only, or a gate on the gate projection w1, might work. n5407's up projection is the right lever.
3. **Use more power.** The commit-rate shift (+.06 to +.08) is reproducible, but an accuracy effect of about 3 pp needs about 300 paired answers. Read passes over long traces now run at about 70 tokens/s, so keep them short: prefixes that end at the decision token, with only the layers you need.
