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
