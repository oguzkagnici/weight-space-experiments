# Notes of editor claude, generation 10

## Diagnosis (free)
- 15 mixed questions in the passk file (64 q x 4 samples, cap 8192). Right vs wrong samples read like genuine uncertainty:
  e.g. mmlupro_2055 ("I'm not familiar with those exact terms"), samples pick C/E/G/G by different guessing heuristics;
  mmlupro_5325 samples split I,I vs E,E (two plausible "first step" options).
- Uncertainty markers in finished answers (214): "not sure/uncertain/hard to tell" in 119 (acc 51.3%) vs 95 without (85.3%);
  "guess" 37 (45.9%) vs 177 (70.6%).
- Read limit is 4,096 tokens per text: 22 of 50 finished mixed-question traces are longer and only their first 4,090 tokens can be read.

## Pass 1 (read, original): 50 finished samples of the 15 mixed questions, first <=4,090 tokens each (162k tokens),
token_logprobs + top-10. Goal: probability of the sampled letter/verdict at commitment points; is gold the runner-up?
- Finalist pairs (code/004): in the second half of finished traces, a two-letter finalist statement ("X or Y", "between X and Y")
  occurs in 20/142 right and 32/72 wrong answers. When gold is in the last stated pair: picked gold 16, picked the other 14
  -> the finalist decision is ~coin-flip (53%). These 14 = 19% of all finished-wrong answers.

## Pass 2 (read, original), queued early: forced-answer probe at the LAST two-finalist statement (after 30% of the trace)
59 texts (passk + 8192 trace bank, prefix <= 4,080 tokens; 58 finished traces; gold in pair 31); suffix "\n\nSo the answer is (";
top-50 at the last position. Question: does the latent preference pick gold more often than the realised pick (16/30)?

## Pass 1 result (code/006)
- Letters at in-thinking commitment phrases ("answer is (", "go with (", ...) are near-deterministic: median p 1.00 in right
  (75) and wrong (44) traces; p<0.9 in 11% / 2%. First commitment letter of wrong traces: p >= 0.97 in 11/12.
- Low-probability sampled tokens (p<0.05): 30.1 per 1k generated tokens in right traces vs 31.0 in wrong ones; mostly
  punctuation and hedges ("maybe", "or", "something").
- => Wrong conclusions are NOT low-probability samples at the decision token. Local sharpening at letter/decision tokens is
  useless. The divergence is spread over the reasoning path.

## Pass 2 result (code/007)
- Forced answer right after the last finalist statement == realised final pick in 51/58 finished traces; letter mass 0.98,
  usually p>0.95. Gold in pair (31): probe picks gold 20, realised 21. Over all 58: probe top == gold 23, realised 26.
- => By the time the model states two finalists, its conclusion is already fixed (latent == final). A commit-at-finalist
  edit cannot change conclusions. The parent's L22 unit (commit after finalist paragraph) is therefore content-neutral, as observed.

## Pass 3 (read, original): forced-answer probes at 15/40/70% of finished traces (mixed questions first), code/008.

## Pass 3 result (code/009): 43 finished traces (42 from mixed questions), forced answer at 15/40/70%
- latent==final: 19/43 (15%), 22/43 (40%), 36/43 (70%); latent==gold: 14, 17, 22; final correct 26/43.
- Wrong traces (17): latent was gold at some checkpoint in 5 (2055 s0/s1, 7456 s0, 5325 s3, 2654 s1), then lost.
  Right traces (26): latent still wrong at 70% in 4, then fixed.
- => The conclusion forms gradually and reasoning improves it on average (latent accuracy 33% -> 40% -> 51% -> final 60%).
  Freezing an earlier belief would lose (-9 at 40%). No single local decision event to repair.

## Hedged commitments (code/010, 011; all finished 8,192 traces, passk + bank, 239 with a commitment)
- FIRST commitment hedged (hedge word within the paragraph/250 chars before it): 66, first letter right 20 (30%), final right 30
  (45%), switched 19; mean 6,259 chars after it. Firm: 173, first right 128 (74%), final 132, switched 8, 1,288 chars after.
- SETTLING (last-letter) commitment hedged: 47, acc 44.7%, median 3,742 chars after; closes within 600 chars in only 9 (44%).
  Firm settling: 192, acc 73.4%; 126 close within 600 chars (81.7%).
- Interpretation: re-checking after a hedged first commitment nets +10/66; the model already re-checks after hedged commits.
  Candidate lever (weak, the only content-directed one left): make the model re-check after a HEDGED commitment instead of
  wrapping up (the parent's L20 unit pushes wrap-up after any stated answer). Expected effect small; test it causally.

## Pass 4 (read, original): 52 finished traces (hedged-commit traces first), <=4,090 tokens each, 149,817 tokens;
inputs of model.layers.23.feed_forward.w1 at all positions + top-10. 125 commitment-break marks, 49 hedged. (code/012)
