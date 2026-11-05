# ed_b, generation 2: report

## Summary
I followed the suggested starting point: the correctness belief at the option positions, read by L22.H12. It turned out to be a negative result.
- H12 already reads all the linearly available correctness signal at the options.
- The heads that write that signal mainly set the model's **confidence**, not which option wins.
- The remaining errors are knowledge errors. Among the wrong options, the model ranks the gold one no better than chance.
- My one candidate is a calibration fix: correct-letter log-prob +0.51 (SE 0.05) on HOLDOUT. Accuracy changes by about zero (-0.59, SE 1.42).

## What the model gets wrong
**Where the answer is chosen.**
- L22.H12 puts 65% of its attention on option END tokens (the last token of each option line) and 12% on the "(X" letter tokens.
- Its attention argmax picks the gold option 40.6% of the time (DEV-160) and agrees with the final answer 72.5% of the time.
- L14.H12 already does the same selection (41.9%).
- H12's attention has no position prior: its argmax counts match the gold counts. The A/B/C-up, I/J-down letter bias arises after selection.

**The correctness belief at the option ends.**
- It forms between layers 8 and 14. Separation along the correct-minus-incorrect direction: h4 .02, h8 .25, h10 .81, h12 .89, h14 1.00.
- Most of it is written by MLP/conv layers.
- The attention heads writing it are L12.H13 (strongest), L10.H4, L8.H1 and L12.H8. L12.H12 writes against it (correlation -0.82 with the rest of the stream).

**No unread signal.**
- Cross-validated probes at the option ends reach 31–41%, no better than H12's own pick.
- On questions the model gets wrong, the probes pick gold 8–16% of the time (chance is about 10%).
- Combining H12 attention, probes, L14.H12 and the final logits gains at most 2 of 160 questions.
- So L22.H12 is not reading the belief weakly. The belief itself is wrong on those questions.

**Letter stage (512 DEV, parent 34.2%).**
- Accuracy read through the output layer at each stage: h22 24.2%, after attn22 34.4%, after mlp22 32.4%, final 34.8%.
- Two letter biases roughly cancel: L22.H12's own late-letter tilt (A -.75, B -1.44, G +1.27) against an early prior (A +1.0, B +1.2) plus mlp22's early tilt.
- The leader-suppressor bank in the L22 MLP is asymmetric, as generation 1 found: A n389 -.76, B n6620/n5859 -1.0, C n1812 -1.7, D–J -3.1 to -4.1.
- A 2-fold cross-validated constant debias would gain only +0.6 points on this set. Rescaling or equalizing any late component in lens screens moves accuracy by at most ±1 point.

**Surface features.** The model picks the longest option 26% of the time (gold 19%). A cross-validated length correction gains nothing. Lexical overlap with the question has no effect.

**Unused knowledge (cloze test, pass 8).**
- I scored each option's text likelihood after the question, on 108 DEV questions.
- It picks gold 25.9% overall and 25% on questions the letter protocol gets wrong (18 of 72; chance about 10%).
- The model therefore holds some knowledge that the letter-selection circuit does not use.
- Adding it to the letter scores (0.25 × z-score) gained +2 of 108 questions, a small sample.

## What I changed and why
**Edits tried.** I tested rescaling L12.H13, the main attention writer of the belief: the head-13 columns (1664:1792) of `model.layers.12.self_attn.o_proj.weight`.

| Edit | Question set | Accuracy change | Log-prob of correct letter | Entropy | Gold rank change |
|---|---|---|---|---|---|
| ×2 (X1) | DEV S1 (256 q) | +1.56 (SE 1.46) | -0.32 (SE .04) | .99 → .90 | +.004 ± .06 |
| ×2 (X1) | HOLDOUT[0:512] | -0.78 (SE 1.00) | -0.31 (SE .03) | | |
| ×0, ablation (X0) | DEV S1 (256 q) | +1.17 (SE 1.70) | +0.56 (SE .08) | .99 → 1.24 | -.02 ± .08 |
| ×0, ablation (X0) | HOLDOUT[0:512] | -0.59 (SE 1.42) | +0.51 (SE .05) | | |

- X0 on HOLDOUT changed 25 answers wrong→right and 28 right→wrong.
- The dose-response is clean: the head sets confidence (entropy, log-prob) and leaves the ranking alone.

**Submitted:** `ed_b_g2_L12H13_ablate` (X0), the better holdout result. It is a calibration improvement; expect about zero change in accuracy. I did not submit X1, which is worse on both measures.

## What I would try next
1. **Use the knowledge in the option text.** Cloze likelihood is right on 25% of the questions the letter protocol misses. Find the circuit that computes how plausible an option's text is at its tokens (possibly the same late MLPs), and see whether it can feed the option-end belief that L22.H12 reads. This is the only lead I found that carries information the model's answer does not already use.
2. **Stop spending effort on letter-bias and confidence edits.** The constant-debias ceiling was +0.6 on my 512-question set. Generation 1's prior removals were negative on the hidden set, and generation 1's suppressor halving was +0.5. Edits of the belief's confidence (this generation) do not change accuracy.
3. Any accuracy edit must improve the **gold rank**. Screen with the paired gold-rank or entropy split (code/026), which separates real ranking gains from sharpening or flattening.
