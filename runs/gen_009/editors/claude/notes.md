# Notes of editor claude, generation 9

# claude, generation 9: notes

Disclosure: as the GA operator I created the holdout bank. I saw its aggregate numbers once (accuracy and cut share), and no per-question rows or traces. I will use it only for the final measurement.

## Bank analysis (code/001, 002; free)
- 13 of 47 cut traces state "answer is (X)" inside the thinking. Most statements are hedged ("Thus likely answer is (D) or H?", "maybe the answer is ..."), at 30-84% of the trace. The first statement is right in 5 of 13.
- Finished traces:
  - 21 with ≥2 statements and an unhedged first statement: it sits at 97% of the trace and always equals the final answer. The L20 wrap-up already closes these.
  - 8 with a hedged first statement: at a median 63% of the trace, ~1,000 tokens before the close. It equals the final answer in 7 of 8 (first right 4, final right 5).
- Size estimate: closing at a hedged single-letter statement could convert up to ~5 of 13 cut traces on the bank and cost ~1 of 8 finished traces. That is about +4 pp if perfectly targeted.
  - Statements naming two letters ("(C) or H?") must be excluded.

## Pass 1 (read, parent): L20 MLP inputs at all positions + top-20 + token logprobs
- Texts: 24 traces with a mid-trace statement (prefix up to the first statement + 600 characters) and 15 full finished traces with a final statement. 69,386 tokens.
- Goal: profile the L20 n5414 gate at hedged vs firm statements, and find a direction for "single-letter hedged statement just completed".

## Pass 1 result (code/004-006)
- **Why the L20 wrap-up unit misses these statements.** It fires at the paragraph break after a statement line. Most mid-trace statements in cut traces do not end their line: "Thus likely answer is (E). Let's examine all options", "... is J? But let's", "... is (I) because ...". At the following newline the gate is far below threshold (z −8 to −14). Where a single-letter statement ends its line, the gate fires: unhedged z +11 to +14 (act ~9), hedged z +5 to +7. Yet the parent still reopens (P(But/We/Let) 0.45–0.65).
- **Correctness of mid-trace statements** (24 bank traces):
  - Cut, single-letter: 10 statements. Right: 2147 (J), 2731 (I), 8198 (C, firm), 2017 (G, cut at the very end). The other 6 are wrong.
  - Finished traces with a hedged first statement: the first equals the final answer in all 6.
  - Two-letter statements ("B or H?"): excluded.
  - Ceiling of "close at a stated single letter": about +3 of 96 on the bank, at almost no cost.
- **Where the cut budget goes** (code/005, all 96 bank traces): cut traces have 49 paragraphs vs 28 in finished ones.
  - Per trace: churn openers 2.6 vs 0.4; "Let's ..." 3.7 vs 1.1; commit paragraphs 1.6 vs 2.9.
  - After a paragraph that ends uncertain, P(churn) is 0.12 vs 0.04. No single category dominates the extra paragraphs.
- **Statement-end tokens are separable at the L20 MLP input.** I used the last token of a single-letter answer statement, including trailing ")." / "?" (53 in 39 texts).
  - Mean-difference direction, leave-one-text-out AUC 0.988.
  - The highest "negatives" are conclusions my regex missed ("Thus answer should be C.", "Thus I'd pick (E).", "Therefore answer likely (E)."), which are real targets. The prompt's instruction tokens reach at most +5 and post-think tokens +15, against a positive median of +32.
- **The parent at statement ends:** P(</think>) is 0.98 where it closes (14 statements, all quoted final statements) and 0.000 at the other 39.
- **Design: Unit A at L23** (no dilution by later layers).
  - Gate: the statement-end direction at the L23 MLP input, minus a threshold.
  - Write: </think> against the reopen words.
  - After </think>, the parent's short-answer units copy the letter.
  - L23 has no other tight duplicates: only ed_a's {331, 881, 5105}; {5821, 2400} is only cos 0.997. So I repurpose a quiet L23 neuron.

## Pass 2 plan: read, parent, same 39 texts (code/007)
- Record the L23 MLP input, L23 MLP output and h24, at all positions.
- Use: build the gate, choose a quiet slot, and estimate the logit change at statement ends and everywhere else (post_ffn_norm + final-norm read-out).

## Pass 2 result and Unit A (code/008-010)
- **L23 MLP input, single-letter statement ends vs all other tokens:** LOTO AUC 0.982, positive median 12.5, negative 99.9th percentile 7.4.
- **Hedged statements score low.** "Thus likely answer is (E).", "J?" and mid-sentence "(I) because" fall below any clean threshold. The direction reads FIRM conclusions, in any wording: "Thus answer should be C.", "Thus I'd pick (E).", "Thus correct answer J.", "Therefore answer likely (E).".
- **Coverage on the bank (code/010).**
  - 8 of 47 cut traces contain a firm conclusion; the first one is right in 5 (8198, 2017, 2147, 10393, 7373).
  - In finished traces the first firm conclusion equals the final answer in 38 of 39; net −1 if closed there.
  - Ceiling: about +4 of 96.
- **Slot.** L23 has no free tight duplicate besides ed_a's group. n5217 is quiet: mean |out| 0.028, max 0.25 over 8,000 tokens; median neuron 0.053.
- **UNIT A (current edit, staged in edited/): L23 n5217 repurposed.**
  - Gate (w1 row) = 2·(d − 8·muh/m). d is the unit mean-difference direction (statement ends minus all other tokens) at the L23 MLP input; muh/m is the mean think-token input, scaled so that c = x·muh/m ≈ 1.
  - Up (w3 row) = muh/m.
  - Write (w2 column) = residual direction raising </think> against the observed continuations (But/Let/We/Hence/Therefore/...). It is taken through the final-norm gain, inverted through the post_ffn_norm gain (regularised), at |col| 1.0, clipped per Q8 block to 0.98× the neighbours' absmax (0.94 kept).
  - Pre-screen (linear read-out of pass-2 recordings):
    - Fires (act > 0.5) on 72% of statement ends. At those, P(</think>) goes 0.000 → 0.999 (median); overall P > 0.5 goes 0.38 → 0.64.
    - Off-target: 72 of 69k tokens change by more than 0.5 logit, all conclusion-like. Nothing changes on prompt tokens (max act 0.05), two-letter statements or post-think tokens.
  - Threshold 6.5 doubles off-target firing for no gain in coverage, so 8 is kept.
- Disk: L20 inputs of pass 1 deleted.

## Passes 3-4 plan: generate, edited only, full bank (paired against the bank = parent)
- Pass 3: B1 = the 8 target cut traces + 40 random bank ids. Pass 4: B2 = the other 48.
- Then passes 5–6: holdout bank (96 ids, 48 per pass) for the final measurement.

## Pass 3 result: Unit A on B1 (48 bank ids, edited only, paired against the bank) (code/012)
| set | parent → edited | diff (SE, z) | +/− | cut | tokens |
|---|---|---|---|---|---|
| targets (8 cut traces with a firm conclusion) | 0 → 4 | +50.0 (18.9, 2.65) | +4/−0 | 8 → 3 | −488 |
| other 40 | 18 → 20 | +5.0 (5.0, 1.00) | +3/−1 | 15 → 16 | −70 |
| all 48 | 18 → 24 | **+12.5 (5.7, 2.21)** | +7/−1 | 23 → 19 | −140 |
- Every new close comes right after a conclusion: "Thus I'll go with J.\n\n</think>", "Thus answer should be (J).", "Thus The answer is (B).".
- Traces diverge early (median at 16% of the trace; 1 of 48 identical). Removing n5217's small old output resamples every trace, so the target gains are partly fresh trajectories that also conclude.
- One loss (5822: the trace diverged at 6% and concluded B instead of J).
- 4 newly cut traces were all wrong before and after.

## Pass 4 result: B2 (the other 48 bank ids); full bank (passes 3+4)
| set | parent → edited | diff (SE, z) | +/− | cut |
|---|---|---|---|---|
| B2 (no targets) | 20 → 22 of 48 | +4.2 (2.9, 1.43) | +2/−0 | 24 → 22 |
| non-target 88 | 38 → 42 | +4.5 (2.8, 1.65) | +5/−1 | 39 → 38 |
| **all 96 bank** | 38 → 46 | **+8.3 (3.2, 2.60)** | **+9/−1** | 47 → 41 |
- Caveat: the gate direction was fit on 39 texts taken from bank traces. The bank is therefore not out of sample for the direction, though it is for the generated outcome. The holdout bank is the measurement.
- The edit is unchanged for the final measurement; no variant was tried (no passes left for one).

## Passes 5-6: holdout bank (96 ids, 48 per pass), Unit A unchanged
- Pass 6 (first attempt, gen_009_claude_p06_2ec58a) FAILED in the server: llama-quantize ran out of disk while building the GGUF. It was not charged. Freed 1.6 GB (old gens' `edited/` dirs that are byte-identical to a saved candidate) and requeued with the same request (work/req_06.json).

## Pass 5 result: holdout H[:48] (code/012 --holdout 5)
| set | parent → edited | diff (SE, z) | +/− | cut | tokens |
|---|---|---|---|---|---|
| holdout 48 | 29 → 29 | **+0.0 (5.2, 0.00)** | +3/−3 | 16 → 15 | +64 |
- Gains:
  - 11722 and 4105: cut → close right after a firm conclusion ("Thus answer should be (I).").
  - 2721: early divergence at 11%; a fresh trajectory reached F.
- Losses:
  - 4235 and 4740: parent right and finished, edited trace newly cut (diverged at 22–27%).
  - 2134: closed on a wrong conclusion ("Thus I'll go with F.").
- Only 3 of 48 traces are identical. The unit perturbs nearly every trace early. Each perturbation is a fresh sample: it rescues some cut traces and breaks some finished ones.
- Reading: the in-bank +8.3 was largely in-sample. On held-out questions the targeted mechanism still works (2 of 2 firm-conclusion rescues), but the gain is about the size of the resampling noise (±3).
