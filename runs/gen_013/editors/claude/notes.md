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

## Pass 2 results (read, original; code/009) — THE DETECTOR IS A SELF-CONSISTENCY SIGNAL, NOT A TRUTH SIGNAL
- Detector = pass-1 corrupt-minus-clean mean direction. Score(substituted) - score(own) at the pick sentence:
  - RIGHT samples, own correct letter replaced by the common wrong letter: h18 +0.58, h20 +0.84, h24 +5.62 (>0 in 25/25).
  - WRONG samples, own wrong letter replaced by GOLD: h18 +0.43, h20 +0.66, h24 +3.79 (>0 in 25/25). The network flags the TRUE answer as the deviation.
  - 9251 s0 hand pair (model's own false lens-sign step vs the corrected step): corrected scores MORE error-like (h18 +0.33, h24 +3.91 ~ 2 sd of synthetic corrupt-clean).
- P(re-check opener next): right samples own 0.50 -> sub 0.79; wrong samples own 0.76 -> gold-sub 0.85; 9251 own false 0.20 -> corrected 0.58.
- Side finding: after the model's own pick statement, P(re-check next) is higher in WRONG samples (0.76) than RIGHT ones (0.50): the model is less sure of its wrong picks, but its re-checks re-derive the same belief (9251).
- Conclusion: the network detects steps that deviate from ITS OWN prediction (surprise / self-consistency), and that already raises re-checking. It has no signal that a confident wrong belief is false. Strengthening detector->re-check should make the model doubt deviations from its beliefs, including correct ones; it cannot catch the errors that make answers wrong (confident beliefs, unchecked cues, knowledge guesses).

## Edit E1 (code/011, args K 4 ALPHA 1 PCT 85): L23 n881 detector -> re-check unit
- n881 folded into near-duplicate n331 (cos .9996; ratio .989); n881 gate = K*(w - T*p)/g_ffn with w = corrupt-minus-clean mean of normalised h22 on TRAIN questions (work/train_qs.json), p = bias proxy, T = 85th pct of train corrupt scores; up = p/g_ffn; w2 col = mean E[Wait, " Wait", Actually, " Actually"] - mean E, Q8-safe (norm 1.10).
- Gate at sentence ends (pass-1 vectors): TEST questions fire corrupt 19% vs clean 4% (train 16% / 1%); AUC .67.
## Pass 3 plan (read, EDITED): held-out (test-question) pass-1 pairs + natural own/sub pairs (pass 2 texts, incl. 9251): P(re-check) and token-logprob off-target changes vs pass 1/2 (parent).

## Pass 3 results (read, EDITED E1; code/013) — gate check FAILS the "clean steps left alone" condition, and doubts true steps
- Held-out pass-1 pairs: P(re-check next) CORRUPT 0.124 -> 0.386 (up in 67%), but CLEAN 0.064 -> 0.233 (up in 49%): leaky, near-global.
- Natural: own correct pick 0.50 -> 0.51; own WRONG pick 0.75 -> 0.77 (unchanged); wrong letter substituted into right samples 0.75 -> 0.95; GOLD substituted into wrong samples 0.88 -> 1.00; 9251 own false step 0.20 -> 0.24 vs corrected step 0.58 -> 1.00.
- => exactly as pass 2 predicts: strengthening the network's own error signal makes it re-check the TRUE step that deviates from its belief, not its own wrong belief.
- Off-target: per-token |dlogprob| mean 0.039, changes from the first tokens of every text (0/212 unchanged), median max |d| 5.6: the gate fires off-target (prompt/sink positions likely).
## Pass 4 plan (read, ORIGINAL): w1 inputs of L23 MLP at ALL positions on held-out pairs + practice traces -> refit gate offline against real background positions (sink projection, threshold for silence), simulate n881 activation exactly.

## Pass 4 results (read, original; L23 w1 inputs, all positions; code/015, 016)
- Calibrated on 14,795 background positions of 8 practice traces, a gate silent on 99.9% of background (T = 99.9th pct) fires on 0 of 12 held-out corrupted sentence ends (and 0 clean). The detector score is high at ALL sentence ends (clean ends at the 71st-99.7th background percentile, corrupt ends a few points higher; mean corrupt-minus-clean after the corruption +0.2 to +5.6 in score units). The corrupt-vs-clean signal is a small paired shift on top of a large "end of sentence" component: there is no distinct, gateable "this step is false" state in the residual stream at L23 input.
- Sink tokens (first 4) projected off: simulated n881 |act| 0 there.
## DECISION
- Self-correction route = NEGATIVE: (1) the error signal is self-consistency (flags deviations from the model's own prediction, incl. true steps), (2) it is too weak to gate (paired only), (3) strengthening it (E1) raises doubt about true steps and leaks.
- Candidate E2 (for the required measurement): refit strict gate (Q .999, K 4; fires on ~0.1% of tokens, the most self-inconsistent points) writing re-check openers. Expected: no change in conclusions.
## Pass 5 plan (generate, EDITED E2, cap 8192, n 4): 6 mixed pass@4 questions with the most finished-wrong samples (6110, 6800, 2731, 5325, 2055, 11805) = 24 answers, paired with the passk bank (parent cached).
## Passes 6-7 queued (generate, EDITED E2, cap 8192): holdout48 ids 0-23 and 24-47 (final measurement).

## BLOCKED (2026-11-16 ~02:10 UTC): passes 5, 6, 7 FAILED (not charged) — llama-quantize Q8_0 failed for lack of disk (free 0.9-1.2 GB during the builds; 4.2 GB after). A gguf build needs ~5.3 GB free (bf16 ~3.5 GB + q8 ~1.85 GB). Budget: 4 used of 8.
- To resume: re-submit work/req_gen5.json (pass@4 mixed, 24 answers), work/req_hold6.json and work/req_hold7.json (holdout48), edit E2 already saved in edited/. Then code/018_paired.py dev 05 ; hold 06 07 (check numbering of the new passes).
