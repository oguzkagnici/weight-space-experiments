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

## Pass 2 result (read, original): ATTENTION at option-judging positions = NULL (code/012, 013)
- 98 prefixes (35 finished samples of the 10 mixed verbal questions; 7 over 4,096 tokens dropped): first evaluation of the gold option (j_gold),
  of the main wrong finalist (j_fin), commit paragraph (j_late). attn_last at all 11 attention layers (sliding-window layers cannot reach the prompt).
- Attention from judging positions to the stem is tiny: ~1% of the mass (stem+cue .010-.014 per head on average; options .11-.14; own trace .77-.80).
  WRONG samples attend slightly MORE to the stem than right ones (j_gold .0113 vs .0096; j_fin .0142 vs .0111; j_late .0068 vs .0043).
- Within question (right - wrong, SE over 10 questions), no head reaches |z| > 3 of 176 at any point (about 9 would pass |z| > 2.26 by chance);
  layer means |z| <= 2.2 (L22 at j_gold +.004, z 2.23; at j_late -0.85). Split-half correlation of the 96 full-attention head differences between
  two disjoint sets of 5 questions: -0.07 (j_gold), +0.06 (j_fin): no replicable head pattern. Only L22.H10 is positive in both halves
  (half B z 2.75 at j_gold, 1.31 at j_fin).
- CONCLUSION OF STEPS 1-2: the stem-check is not a mechanism that fails in wrong samples. Wrong samples see the cue (text), attend to the stem as
  much or more (network), and quoting the cue in front of the conclusion does not change it (causal). The failure is knowing what the cue implies.
  Per the gate rule I stop spending passes on this line.

## EDIT (minimal, to have a measured candidate; expected null): L22.H10 output x1.5
- model.layers.22.self_attn.o_proj.weight columns 1280:1408 (head 10's 128 inputs) x1.5 (code/014). Rationale: the only head whose right>wrong stem
  attention replicates on both halves. Nothing else changed.
- Pass 3 (read, edited): the identical 120 texts of pass 1 (paired): P(gold) at the probe with none/ctrl/cue insertions, and whether the edit
  enlarges the cue effect. Then passes 4-5: HOLDOUT48 (ids 0-23, 24-47) at 8,192. Passes 6-8 stay unused (the line is closed; saves queue time).

## Pass 3 result (read, edited = L22.H10 x1.5, paired with pass 1; code/016): NULL as expected
- dP(gold) at the probe, edited - original, all |dP| <= .0033 (SE .0015-.0035) for none/ctrl/cue at both cuts, also on the 10 cue questions not used
  to pick the head; top-1 = gold unchanged (5->4, 1->1, 3->3, 0->0, 1->1); the cue effect is not enlarged. Letter mass unchanged (+.01).
- The edit is the submitted candidate only because a measured candidate is required; passes 4-5 (holdout48) are already queued.

## Passes 4-5 result (HOLDOUT48 at 8,192, edited = L22.H10 x1.5; code/015)
- overall 62.5 -> 60.4 (-2.08, SE 6.31, W->R 4 / R->W 5); both-finish (38) 73.7 -> 71.1 (-2.63, SE 5.95, +2/-3); cuts 7 -> 8; 0/48 identical,
  first difference at a median ~2% of the trace (resampling).
- SUBMITTED claude_g14_stemcheck_L22H10_amp (code/017). Report: report.md. Passes used 5/8 (3 left unused: the line is closed).

## REPORT
(report.md could not be written by this agent; this is the report.)
Editor `claude`, generation 14 -- THE STEM CHECK IS NOT THE BOTTLENECK (negative result).
Candidate `claude_g14_stemcheck_L22H10_amp` (minimal, submitted because a measured candidate is required): L22 head 10 output x1.5
(model.layers.22.self_attn.o_proj.weight cols 1280:1408). HOLDOUT48 at 8,192: overall -2.08 (SE 6.31, +4/-5); both-finish (38) -2.63 (SE 5.95,
+2/-3); 0/48 traces identical (resampling). Passes 5/8 (3 read, 2 generate); 3 left unused because the read gate closed the line.
Material: only 10 mixed verbal questions exist in the pass@4 bank (>= 40 not available without generate passes); I also used all 31 finished-wrong
verbal questions (72 samples). Hand-labelled decisive stem cues: 15 of 31 have one; the other ~16 are pure recall (no stem check can help them).
1. TEXT: wrong samples do NOT skip the cue. Every wrong sample uses the decisive cue during judging, usually many times (6110 "informed" 18-50x,
   1409 "WEAKEST" 20x, 6614 "weight/BMI" 37-62x, 1643 "after judgment" 5-9x in all 4); right samples mention it only ~20-30% more. They draw the
   wrong inference from a cue they see: a knowledge gap (informed -> open-label; newly discovered evidence must exist at trial).
2. NETWORK: stem attention from judging positions is ~1% of the mass and slightly HIGHER in wrong samples (.0113 vs .0096 at the gold option);
   no head |z| > 3 of 176 within question; split-half correlation of head differences -0.07 / +0.06.
3. CAUSAL GATE: quoting the cue right before "The answer is (" vs quoting a non-decisive stem sentence: P(gold) +0.041 (SE .039) early,
   +0.085 (SE .066) late; vs no insertion -0.001 (SE .066); top-1 -> gold +2/32, +1/12. The conclusion does not move.
Edit read gate (pass 3, paired): |dP(gold)| <= .0033 everywhere, top-1 unchanged.
NEXT: stop targeting how the model reads the question. Gens 13-14 show steps are locally right, the error signal tracks the model's beliefs, the
stem cues are seen and used, and re-stating them does not help. Wrong verbal answers are missing/wrong domain knowledge. A next test: teacher-force
the cue's IMPLICATION (e.g. "everyone knows they get the drug, so it is not blinded") and measure P(gold): if the model follows supplied knowledge,
the bottleneck is recall (MLP memories), which is a per-fact knowledge-editing problem that is unlikely to move many questions at once.
Do not use dev sets picked from parent failures.
