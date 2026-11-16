# Notes of editor claude, generation 14


## Plan (before any pass)
Target: STEM-CHECK CIRCUIT. Material: pass@4 bank at 8,192 (64 q x 4) + 32 bank-only questions (sample 0).
- Inventory (code/001-002): only 12 mixed questions in the pass@4 bank, 10 verbal (10964, 1171, 2055, 2654, 2731, 5325, 6110, 6562, 6800, 7456).
  The ">= 40 mixed verbal questions" target is not reachable from the banks; generating more mixed samples costs 24 answers per pass at 8,192.
  I use the 10 mixed verbal questions for the within-question attention comparison and the 31 finished-wrong verbal questions (72 samples) for the rest.
- Hand-labelled the decisive stem cue (code/005): of 31 finished-wrong verbal questions, 15 have a stem detail that decides between gold and the
  wrong pick (6110 "informed", 2654 "dreamed he was falling", 6562 smoker -> non-selective beta-blocker, 6800 maxillary sinus, 1409 "WEAKEST", ...);
  the other ~16 are pure recall (Carens, Salmond quote, Tunisia 2018 %, theory names, definitions): no stem check can help them.
- Pass 1 (read, original; code/006-007): CAUSAL INSERTION GATE. 32 finished-wrong samples of the 15 cue questions, cut at ~30% of the trace (early)
  and, for one wrong sample per question, at the start of the final commit paragraph (late). Insert
  '\n\nLet me re-read the key part of the question: "<cue>"' vs the same frame quoting a NON-decisive stem sentence of similar length (control),
  then the probe '\n\nThe answer is ('. Measure log P(gold) - log P(wrong pick) at the probe (top-50 at the last token). 94 texts, 184k tokens.
- Pass 2 (read, original; code/008): ATTENTION at option-judging positions in the 10 mixed questions (35 finished samples): prefixes ending at the first
  evaluation of the gold option, of the main wrong finalist, and at the commit paragraph; attn_last for all 11 attention layers; mass on stem tokens,
  on the cue span, on option lines, per head; right vs wrong within question. 105 texts, 173k tokens.
- Only if both are positive: localise heads (patch/ablate via edited weights, read pass) and strengthen where they act; else report the negative result.

## Step 1(a), text level (free; code/009, 010) -- the "wrong samples skip the cue" hypothesis is FALSE at the text level
- Cue content words (code/009) appear after the first 20% of the trace in 97% (wrong, n 32) vs 100% (right, n 9) of finished samples of the 15 cue
  questions; within the 5 mixed cue questions the rate per 1,000 characters differs by +0.50 (SE 0.58).
- Decisive keyword counts (code/010), per finished sample after 20% of the trace: EVERY wrong sample uses the decisive cue, usually many times:
  6110 "informed/aware/open-label" 18-50x in the 3 wrong samples (right sample 48x); 1409 "weakest" 20x (wrong); 6614 "weight/obese/BMI" 37-62x
  (all 3 wrong); 6562 "smok/COPD/bronch" 53x wrong vs 92x right; 6800 "maxillary/transillum/purulent" 32-74x wrong vs 48-54x right;
  2654 "dream" 63x wrong vs 28-61x right; 1643 "after judgment/did not exist" 5-9x in all 4 wrong; 1196 "extended/30 days" 6-15x in all 4 wrong.
  Within the 5 mixed cue questions the right samples use it somewhat more per character in 5/5 (about +20-30%), but the wrong ones do not skip it.
- Reading: wrong samples SEE the cue and reason about it; they draw the wrong inference from it (they do not know, e.g., that informing every patient
  makes the trial open-label, that newly discovered evidence must exist at trial time, that a 30-day TRO extension is appealable as an injunction).
  This is a knowledge/inference failure, not an attention failure. Passes 1-2 test whether the network itself agrees.

## Pass 1 result (read, original): CAUSAL INSERTION GATE = NULL (code/011)
- 32 finished-wrong samples of the 15 cue questions, early cut (~30% of the trace), 3 variants; 12 of them also at the late cut (commit paragraph).
- Bounded statistic, P(gold) among the 10 letters at the probe "The answer is (" (SE by question):
  early cue - ctrl (non-decisive stem quote) +0.041 (SE 0.039); early cue - none -0.001 (SE 0.066); late cue - ctrl +0.085 (SE 0.066).
  Top-1 letter becomes gold: early none 5/32, ctrl 1/32, cue 3/32 (net vs ctrl +2, vs none -2); late ctrl 0/12, cue 1/12.
  Log-margin gold - pick, cue - ctrl: early median +0.51 (up in 21/32), late median +0.42 (6/12); the means are dominated by letters outside the
  top-50 (floor -30), so I use the bounded statistic.
- Reading: putting the decisive cue right in front of the conclusion does not turn wrong conclusions right (at most +0.04-0.09 P(gold), within
  noise). With the text-level result (wrong samples already use the cue many times), the stem check is NOT the bottleneck: the model sees the cue
  and does not know what it implies. By the gate rule, a weight edit that makes the model attend to / re-state the cue cannot help.
