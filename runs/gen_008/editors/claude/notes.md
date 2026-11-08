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

## Pass 2 result (read, stack + unit vs pass 1) -> code/005
- At positives: P(" I"/" answer") 0.096 -> 0.779, P(" we") 0.361 -> 0.029; " I"/" answer" top-1 in 77% (the rest are
  positives where the gate stays below threshold, e.g. 1275, 1740-Thus: unchanged).
- Off-target: mean |dlogp| 0.00021 over 99,534 teacher-forced tokens (gen 6 L21: 0.004), p99 0.0015, 22% exactly 0.
  Largest changes are near-target firings (end of "Thus answer letter B or D." -> " Let" -1.08 .. -0.38).
- Passes 3-4: generate edited (stack + unit) on the 96 bank questions (same split as gen 7: DEV32+EXTRA16, REST48),
  paired against the bank; also compare with gen-7 claude pass 3-4 (L22 gate alone) on the same questions.

## Pass 3 result (generate, stack + unit, DEV32 + EXTRA16, paired vs bank) -> code/006
- ALL 48: parent 18 -> 25 (+14.6 pp, SE 6.7, z 2.19), +9/-2, cut 27 -> 19, byte-identical 0/48.
  DEV32 +18.8 (+8/-2, cut 21 -> 15); EXTRA16 +6.2 (+1/-0).
- CAVEAT: all 22 positive traces the opener direction was fitted on are DEV32 questions -> DEV32 is in-sample.
  EXTRA16 and REST48 have no overlap with the fit.
- Two-finalist commits in the edited traces: 10 (bank 3); two-finalist paragraphs 21 (bank 39).
- Reference, same 48, gen-7 claude pass 3 (L22 gate alone): -4.2 pp (+1/-3, cut 27 -> 27).
  The gen-8 run also contains ed_a's L23 short answer (carry), so this is stack + unit vs parent, not the unit alone.
- Pass 4 (REST48, out of sample) is the honest test; gen-7 L22 alone gave +6.2 there (+3/-0).

## Pass 4 result (generate, stack + unit, REST48 — fully out of sample)
- REST48: parent 19 -> 24 (+10.4 pp, SE 4.5, z 2.34), +5/-0, cut 26 -> 24. Gen-7 L22 alone on the same 48: +6.2 (+3/-0).
- All 96 (passes 3+4): 37 -> 49 (+12.5 pp, SE 4.0, z 3.13), +14/-2, cut 53 -> 43.
- Gains in REST48 (2017, 4192, 6014, 7261, 5822): each edited trace has an early "Thus answer (likely) X" / "Thus I think
  answer is X" that the bank trace lacks, then finishes. 3 of 5 were cut in the bank.
- Not yet attributable: the run contains carry (L22 gate + ed_a L23 short answer) + my unit.

## Pass 5 (submitted): carry alone, generate, REST48
- edited/ was temporarily replaced by the carry tensors for the submit (snapshot verified = carry, 6/6 files), then the unit
  tensors were restored (verified 6/6; backup in work/unit_edited/). Paired carry vs stack+unit on REST48 = the unit's own effect.

## Passes 5-6 result (carry alone, generate) -> code/007
- stack+unit vs carry: ALL 96 +11.5 (SE 3.9, z 2.95) +13/-2; REST48 +8.3 (SE 4.0, z 2.07) +4/-0; EXTRA16 +12.5;
  DEV32 +15.6 (in-sample). 14/96 identical to carry.
- carry vs parent on the bank: +1.0 (SE 3.5), +6/-5, cut 53 -> 47.
- Submitted `claude_g8_stack_commit_opener_L23` (code/008). Report in report.md.
