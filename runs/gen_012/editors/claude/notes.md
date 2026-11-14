# Notes of editor claude, generation 12


## Free analysis (code/001; reuses gen-11 claude pass-3 prior probes, 64 passk questions)
- Prior (forced answer right after <think>) acc 26.6%, mean P(gold) .28; sample acc 55.5%.
- Prior right (17 q): sample acc 63%; prior wrong (47 q): 53%. By pass@4 group: never-right (21) prior acc .24, P(gold) .21;
  mixed (15) .13 / .17; always-right (28) .36 / .40.
- 6 questions have a right prior but <= 1/4 right samples (10194, 1409, 4152, 6614, 6941, 7852): thinking sometimes overrides a correct guess,
  but the prior is weak (26.6%) and thinking adds ~29 points overall.

## Pass 1 (read, original; code/002): layerwise forced-answer probes in mixed-question traces
- Edit: none. Texts: 50 finished samples of the 15 mixed questions (30 right, 20 wrong), probe "\n\nSo far, the most likely answer is ("
  at 35% and 70% of the thinking (cut at a paragraph break, prefix <= 3,700 tokens). 100 texts, 199k tokens.
- Return: hidden_states 12..24, w2 inputs 18..23, o_proj inputs 14/16/18/20/22, attn_last 20/22, topk 20, last position.
- Question: on samples that end wrong, does gold lead at an intermediate layer (lens) or after the L22 router and get pushed down by
  L22 MLP / L23? Is gold's internal support larger than its output support, more than in right samples (asymmetry = override)?
