# Notes of editor claude, generation 8


## Plan (claude, gen 8)
- Starting edit: the carried stack (gen-7 ed_a L23 short answer + gen-7 claude L22 two-finalist gate), copied into edited/.
- Observation (gen-7 passes 3-4, edited): after the gate's commit opener, the model usually keeps deliberating
  ("Hence we need to decide ...", "Thus maybe ...", "Hence examine ..."). The two traces that went on with "Thus I think E
  is best" / "Thus I'll choose (A)" finished. Hypothesis: a second gated unit at the opener after a two-finalist paragraph,
  writing " I" (-> "I'll choose / I think X") minus deliberation continuations, completes the decision; then the parent's
  L20 wrap-up neuron (fires after a stated answer) closes the thinking.
- Host: a freed L23 duplicate slot (ed_a's fold: 881 -> {331, 5105}, cos .9995), cheaper than a quiet-neuron removal.
- Pass 1 (read, edited = stack): 22 parent traces cut at their first two-finalist break + "Thus"/"So" (44 texts),
  plus 10 full finished traces; L23 MLP input, topk 20, all positions (99,588 tokens). code/001.

## Pass 1 result (read, stack) -> code/002
- After "Thus"/"So" opening the paragraph that follows a two-finalist statement, the stacked model plans " we" (15.8 of
  44 expected mass; "Thus we need to decide"), " I" 2.8, "," 2.8, " which"/" let" 1.8.
- Direction at the L23 MLP input (positives minus 103 paragraph-initial openers elsewhere): LOTO AUC 0.963.
  thr 6.0 (bias proxy): positives 75%, other openers 1.0%, all tokens 0.09%. Prompt-sink tokens below threshold (and projected off).
- Fold (code/003): h_881 = 1.139 h_331 - 0.158 h_5105 (residual 1.4% of n881's output); w2 columns folded without clip.
- Unit (code/004 6.0 0.5 4 2): L23 n881, w1 row norm 6.1, w3 = mu_hat/mp, w2 col norm 1.43 writing " I"/" answer" minus
  " we"/" let"/" which"/" option"/" what"/" list"/" maybe"/" need". In-sample: positives median pre-activation +12.9,
  11 of 33,185 other tokens > 0, prompt-sink max -9.9.
- Pass 2: read, edited (stack + unit), same 54 texts, topk 20 + token logprobs: the exact effect of the unit vs pass 1.
