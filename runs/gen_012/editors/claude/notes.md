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

## Pass 1 result (code/003-005): NO latent gold anywhere at the forced-answer readout (H2 refuted for mixed questions)
- Letter lens (own-RMS final norm, 10 letters), argmax==gold: right samples h15 .43/.67 (35%/70% probes) ... h24 .57/.80;
  wrong samples h12-h24 <= .15 at every layer. In thinking mode the stated belief is readable from h15 (after L14) on - unlike direct
  mode, where letters only appear at L22.
- Wrong-sample probes whose output is not gold (35): gold is the lens argmax at NO layer 16-24 (0/35 at each); gold above the output's
  top letter at <= 3/35 layers-probes. Right-sample probes: other letters top the lens at h16-h22 in 9-23/41, i.e. intermediate layers are
  noisier, never "better".
- DLA (frozen final scale, centered letters) to the selected letter: L22 H12 +8.44, L22 MLP -3.76 (leader suppression), L22 H15 +1.43.
  The only positive gold-minus-top terms in wrong probes are the generic negative-feedback components (L22 MLP +2.64, L22 H11/13/14 +0.5).
- H12 attention at the probes (code/005): ~50% prompt / 50% trace at 35%, 33/67 at 70%. Option lines get 8-10%; in WRONG samples the
  PICK's option line gets 3-5x the gold line (0.024 vs 0.005); trace letter mentions: pick .11/.20 vs gold .04/.05. In right samples gold=pick.
  The query built from the trace selects the stated belief; nothing points at gold.
- => The readout is faithful to the trace's stated belief at every layer; "knows but does not say" does not occur at the answer readout.
## Free checks (code/006-009)
- Negation questions: 6% of practice; no lever. Arithmetic statement checker: not informative (wrong traces have few computations).
- Stated first hypotheses are late (median 99% of the thinking = the conclusion); == prior in 30%, no worse than others.
- Prior-right-but-lost questions (6): mostly lost by CUTS (10194 3/4 cut, 4152 4/4, 1409 3/4), one "two options both seem correct"
  tie-break (6941 E vs J), one knowledge error (7852). Not a content lever.

## Pass 2 (read, original; code/010): generalise H2 + prior statistics
- Edit: none. Texts: 55 probes in 28 finished WRONG traces of 13 non-mixed questions (50% and end of thinking), 47 right-trace probes at 50%,
  320 pre-thinking prior probes (render_think + "So far, the most likely answer is (") on practice questions outside banks/holdouts.
  hidden_states 14,15,16,18,20,22,23,24 + topk 20, last position.
