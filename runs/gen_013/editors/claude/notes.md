# Notes of editor claude, generation 13

# Gen 13 `claude` notes — target: SELF-CORRECTION

## Pre-pass findings (text only, no passes)
- code/001: natural arithmetic slips are ~absent in practice traces: 1,134 checkable "a op b = c" statements (strict regex), 7 flagged wrong, most are regex artefacts; the real ones (e.g. 15.30+1.7136=16.9136, 100000/49=1020408) are followed by a re-check within 400 chars (6/7). Arithmetic self-correction already works; MMLU errors are factual/judgement.

## Pass 1 (read, original) — plan
- code/002: 140 matched pairs from the parent's RIGHT finished practice traces (mixed pass@4 questions excluded): 100 antonym/negation swaps in one sentence, 40 arithmetic results changed. Text ends after the step sentence + its original separator. Returns: token logprobs, top-50 next-token, hidden states last position L8-L24.
- Questions: will measure P(re-check opener) clean vs corrupt; surprisal of the swapped token; a linear probe fit on half the questions, tested on the other half.
- Edit: none (original).

## Hand classification of finished-WRONG samples on mixed pass@4 questions (text only; first wrong step vs right samples)
- 9251 s0 (physics, lens): real STEP error: mixes 1/f = 1/v + 1/u with Cartesian u = -50 -> 4 D. It hesitated ("1/f = 1/v - 1/|u|? Actually") and then "double-checked" by re-deriving with the same convention: the check repeats the error, it does not detect it.
- 6800 s1/s2 (health): FALSE FACT asserted (Mycoplasma for purulent sinusitis; "S. pneumoniae: similar but more severe"); right samples check each option against the purulent discharge cue.
- 6110 s0-s2 (health): decisive cue ignored ("patients are informed" -> open-label); wrong samples pick the generic "clinical trial". Right sample s3 checks the cue.
- 2654 s1 (psych): picks sleep paralysis though the stem lacks immobility; right samples check that cue.
- 5325, 2055, 10964, 1171, 2731: KNOWLEDGE GUESSES among plausible options (right samples also guess: "my best guess: H").
- So of 10 mixed questions, ~4 have an identifiable wrong step, and in all of them the wrong step is a confident, high-probability belief or an unchecked cue, not a slip the model wrote against its own prediction.

## Pass 1 results (read, original; code/003, 006)
- Corrupted token is very surprising: median logprob clean -0.58 vs corrupt -13.3 (swaps), -0.00 vs -20.8 (arith); corrupt lower in 100%.
- Next-token re-check openers (Wait/Actually/Hmm/Hold/No/Let/Check/Double): swaps 0.049 -> 0.077 (x1.6, up in 69% of pairs); arith 0.001 -> 0.065 (mean; up in only 40% of pairs: a few strong).
- Corrupt-vs-clean direction (diff of means, fit on half the questions): held-out PAIRED accuracy 0.83-0.90 at h8-h24 (h14 swap .90). Transfers across kinds (arith->swap .84 at h14/h24, swap->arith .70-.82). UNPAIRED AUC only 0.63-0.71 (direction small vs context variance; a gate on it would be leaky). Correlates with surprisal (r -0.2..-0.36 with logprob diff): partly a surprise signal.
- So: the network registers a step that is improbable under its own prediction, and re-checking rises somewhat.

## Pass 2 plan (read, original) — code/008
- Own-pick statement vs substituted letter at the first sentence where a pass@4 sample states its pick (25 wrong samples: substitute gold; 25 right samples: substitute the common wrong pick), + hand pair 9251 s0 (model's false lens-sign step vs corrected step).
- Prediction if detector = self-consistency: in wrong samples the GOLD substitution looks MORE error-like than the model's own wrong pick (no truth signal). Edit: none.
