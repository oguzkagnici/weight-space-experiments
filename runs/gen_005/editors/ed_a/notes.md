# Notes of editor ed_a, generation 5

## Split
work/split.json: trace bank 96 -> DEV 64 (35 cut, acc .375) / HOLDOUT 32 (17 cut, acc .406), seed 5, stratified by cut.
HOLDOUT traces are not used for any analysis.

## Diagnosis on the trace bank (no passes)
- code/001: verbatim re-copying of the option list is NOT cut-specific (char share 5% cut vs 9% finished).
- code/002/003: repeated option-by-option sweeps: 2.4 per cut trace vs 1.2 per finished, but per 1k tokens the
  same (0.48 vs 0.49 opener lines/1k). Re-sweeps are a symptom of length, not a discriminating behaviour.
- code/004: only 13/52 cut traces ever state a parenthesised letter leaning (first leaning = gold 5/13).
- code/006 (DEV, paragraph openers per 1k tokens, finished vs cut):
  "Alternatively(, maybe)" 0.02 vs 0.26 (1 vs 28 occurrences), "Thus we (need)" 0.04 vs 0.28, "But perhaps" 0 vs 0.10,
  "Better approach" 0.02 vs 0.09, "We can" 0.02 vs 0.14, "Given the" 0.09 vs 0.30, "We need" 0.32 vs 0.61,
  "Let's examine" 0.17 vs 0.33. Commit openers the other way: "Thus answer" .75 vs .15, "Thus final" .21 vs .01.
- Qualitative: two stuck styles. (1) knowledge questions (law/health/philosophy/psych): hypothesis churn: a paragraph
  ends "Not sure."/"Hmm." and the next opens "Alternatively, maybe <new hypothesis>", or a new full option sweep.
  (2) physics/engineering: method thrashing ("Better approach", "Another approach", "Alternatively, maybe they intend").
- Key asymmetry under det1: a cut trace scores 0, so an edit that fires ONLY in states that occur almost only in
  traces that will be cut cannot lose; it can only rescue. The hypothesis-churn restart ("Alternatively" after an
  uncertainty statement) is such a state (28 vs 1 in DEV).

## Plan
Pass 1 (read, original): 64 DEV traces (trace_text), positions all, token_logprobs, topk 20, inputs of
model.layers.20.feed_forward.w1 and model.layers.23.feed_forward.w1. ~168k tokens.
Goal: next-token distribution at every paragraph break; a "churn restart" direction at MLP inputs; check it is
silent on finished traces; find a quiet neuron to repurpose.
