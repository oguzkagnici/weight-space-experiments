# Notes of editor claude, generation 13

# Gen 13 `claude` notes — target: SELF-CORRECTION

## Pre-pass findings (text only, no passes)
- code/001: natural arithmetic slips are ~absent in practice traces: 1,134 checkable "a op b = c" statements (strict regex), 7 flagged wrong, most are regex artefacts; the real ones (e.g. 15.30+1.7136=16.9136, 100000/49=1020408) are followed by a re-check within 400 chars (6/7). Arithmetic self-correction already works; MMLU errors are factual/judgement.

## Pass 1 (read, original) — plan
- code/002: 140 matched pairs from the parent's RIGHT finished practice traces (mixed pass@4 questions excluded): 100 antonym/negation swaps in one sentence, 40 arithmetic results changed. Text ends after the step sentence + its original separator. Returns: token logprobs, top-50 next-token, hidden states last position L8-L24.
- Questions: will measure P(re-check opener) clean vs corrupt; surprisal of the swapped token; a linear probe fit on half the questions, tested on the other half.
- Edit: none (original).
