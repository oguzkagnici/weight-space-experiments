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
