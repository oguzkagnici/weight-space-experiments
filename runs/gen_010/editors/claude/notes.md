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

## Pass 2 result (code/007)
- Forced answer right after the last finalist statement == realised final pick in 51/58 finished traces; letter mass 0.98,
  usually p>0.95. Gold in pair (31): probe picks gold 20, realised 21. Over all 58: probe top == gold 23, realised 26.
- => By the time the model states two finalists, its conclusion is already fixed (latent == final). A commit-at-finalist
  edit cannot change conclusions. The parent's L22 unit (commit after finalist paragraph) is therefore content-neutral, as observed.

## Pass 3 (read, original): forced-answer probes at 15/40/70% of finished traces (mixed questions first), code/008.

## Pass 3 result (code/009): 43 finished traces (42 from mixed questions), forced answer at 15/40/70%
- latent==final: 19/43 (15%), 22/43 (40%), 36/43 (70%); latent==gold: 14, 17, 22; final correct 26/43.
- Wrong traces (17): latent was gold at some checkpoint in 5 (2055 s0/s1, 7456 s0, 5325 s3, 2654 s1), then lost.
  Right traces (26): latent still wrong at 70% in 4, then fixed.
- => The conclusion forms gradually and reasoning improves it on average (latent accuracy 33% -> 40% -> 51% -> final 60%).
  Freezing an earlier belief would lose (-9 at 40%). No single local decision event to repair.

## Hedged commitments (code/010, 011; all finished 8,192 traces, passk + bank, 239 with a commitment)
- FIRST commitment hedged (hedge word within the paragraph/250 chars before it): 66, first letter right 20 (30%), final right 30
  (45%), switched 19; mean 6,259 chars after it. Firm: 173, first right 128 (74%), final 132, switched 8, 1,288 chars after.
- SETTLING (last-letter) commitment hedged: 47, acc 44.7%, median 3,742 chars after; closes within 600 chars in only 9 (44%).
  Firm settling: 192, acc 73.4%; 126 close within 600 chars (81.7%).
- Interpretation: re-checking after a hedged first commitment nets +10/66; the model already re-checks after hedged commits.
  Candidate lever (weak, the only content-directed one left): make the model re-check after a HEDGED commitment instead of
  wrapping up (the parent's L20 unit pushes wrap-up after any stated answer). Expected effect small; test it causally.

## Pass 4 (read, original): 52 finished traces (hedged-commit traces first), <=4,090 tokens each, 149,817 tokens;
inputs of model.layers.23.feed_forward.w1 at all positions + top-10. 125 commitment-break marks, 49 hedged. (code/012)

## Gate (code/013, 014)
- Mean-difference direction at L23 MLP input: hedged vs firm commitment breaks LOTO AUC 0.81, but >1% of other tokens score
  above the hedged median. Shrinkage LDA, 8-fold grouped CV: AUC hedged vs firm 0.76, hedged vs other 0.987; at a threshold
  keeping 50% of hedged breaks, 14.7% of firm breaks and 0.15% of other tokens fire. Parent's top-1 opener at hedged breaks:
  But 11, Thus 9, Let 6, Wait 5, Alternatively 3, Hence 2 (of 49) -> it already re-checks about half the time.
## Current edit (code/015, K=1, ALPHA=0.2, PCT=60): L23 only
- fold n881 into n331: w2[:,331] += 0.9893*w2[:,881] (cos >= 0.9996).
- n881 gate w1 = K*(w_lda - T*p), w_lda = shrinkage LDA (hedged breaks vs firm+other, all pass-4 texts), projected off the
  first-4-prompt-token subspace; p = min-variance bias proxy (x.p mean 1.0, sd 0.022); T = 60th pct of hedged scores.
  In-sample fires(>0): hedged 43%, firm 8%, other 0.02%; prompt sink max pre-act -44.
- w3 = 0.2*p; w2 col = g_final*(mean E[But,Wait,Let,Alternatively] - mean E[Thus,Hence,</think>,Therefore,So])/g_post,
  scaled so 90% of entries fit 0.98 x Q8 block absmax, rest clipped (|col| 1.05).
## Pass 5 (read, edited): 21-ish hedged-heavy pass-4 texts (<=75k tokens), topk 10 all positions -> firing/effect/off-target.

## Pass 5 result (code/016; in-sample texts): hedged breaks (39) median P(reopen openers) 0.359 -> 0.566, P(wrap) 0.039 -> 0.006,
reopen +0.2 in 10/39; firm breaks (35) unchanged (0.271/0.348); off-target top-1 changed at 0.042% of 69,425 positions,
mean |d top-1 logprob| 0.00038 (p99 0.0015). Mechanism acts as designed; off-target change is small but will still resample traces.
## Final measurement: passes 6-7 = holdout48 at 8,192 (edited only, 24+24); pass 8 = DEV, 6 passk questions with most hedged
commitments x 4 samples at 8,192 (edited only; parent cached).

## BLOCKED (2026-11-12 ~12:20 UTC): disk full
- Passes 6 (holdout48 x2) and DEV failed at llama-quantize (disk: 719 MB free; a GGUF build needs ~5.3 GB). Failed passes do
  not count: budget 5 used of 8. The parent's selection pre-runs gen_010_prerun_parent_00..05 failed the same way.
- Freed my own activation dump (work/fwd_04/in_model...w1.npy, 600 MB) -> 1.3 GB free; still not enough.
- The edit (code/015, K=1, ALPHA=0.2) stays staged in edited/. Requests ready: work/req_hold1.json, req_hold2.json, req_dev.json.
