# Notes of editor ed_f, generation 12


## 0. Free diagnosis (code/001-003; passk + 8,192 bank, no passes)
- Wrong FINISHED answers sit in verbal subjects (passk: law 20, health 11, other 10, economics 9, psychology 9; physics 1, math 2,
  business 0, chemistry 0). Quantitative subjects are right or cut.
- Content-only headroom from trajectory variance is small: on the 60 passk questions with a finished sample, mean finished accuracy
  62.2%, majority-of-finished 65.3% (+3.1). The +5.7 "maj@4" figure in TASK.md includes cut-avoidance.
- The 17 never-right questions with finished samples give the SAME wrong letter in most samples (IIII, HHHH, EEEE, CCCC, AAAA, BBBB):
  stable wrong beliefs, not slips. Reading 9 of them: plain knowledge/judgement errors (law doctrine, attribution, survey numbers).
  No recurring negation / mapping slip seen.

## 1. Choice of target and justification (written before any pass)
Four gated, behaviour-specific edits (gens 10-11) were verified mechanistically and moved nothing: they act on one token class and the
latent belief behind the text stays the same. What is left is the belief itself, i.e. how well the network retrieves what it knows on
EVERY token. So I pick a target that (i) acts on the latent belief rather than on a visible habit, (ii) can be checked with the gate rule
on many probes cheaply.
TARGET (TASK item "latent knowledge present but lost before the answer; any mechanism whose repair raises P(gold)"):
  H1: in the late MLPs, part of the write to the residual stream is low-energy, high-order "noise" (small singular directions of the
  down-projection w2 / of w1,w3) that carries no option-specific knowledge but perturbs the letter read-out and competes with the
  knowledge-bearing leading directions. (Rank reduction of late MLP matrices is known to raise factual accuracy in other LMs.) If so,
  removing these directions in a specific late matrix raises P(gold) at forced-answer probes.
  Test plan:
   pass 1 (read, original): forced "So the answer is (" probes after thinking prefixes of bank traces (held-out split A/B by question),
     recording hidden states and all MLP w2-inputs at the probe token. Offline (linear read-out, exact for L23): effect on the gold-letter
     logit / gold rank of dropping groups of singular directions of each late w2; also per-component attribution to find WHERE (which
     layer, which part of the spectrum) anti-gold writes sit. Split A chooses, split B confirms.
   pass 2 (read, edited): same probes -> measured P(gold) / gold rank change on split B (THE GATE). If it does not rise: stop, try
     the next best located component (pass 3-4), and do not spend generate passes on an edit that fails the gate.
   passes for final measurement: holdout48 at 8,192 (2 generate passes), plus one DEV generate if a pass is left.
  Alternatives considered and why not: (a) re-check after hedged commitment (gen 10: mechanism did not survive generation; both-finish 0);
  (b) first-in-list source (gen 11: P(gold|pair) did not move even with the circuit cut 42%); (c) prompt-attention boost (would pull
  toward the no-thinking prior, which is gold for only 2/15 mixed questions). Expected effect of H1: small; prior of passing selection low.

## Pass 1 (read, original; no edit staged): work/req_p1.json (code/004)
Probes = prompt + thinking prefix cut at a paragraph break + "\n\nSo the answer is (". 96 bank traces (8,192 bank, sample 0) at <=40%
of the thinking (<=1,900 tokens) + passk samples 1 and 2 at <=15% (as many as fit in 196k tokens). Split A/B by md5(question id).
Recorded at the last token: hidden states 0-24, MLP w2 inputs of all layers, top-20.

## Pass 1 result (read, original, 160 probes = first 160 of work/probes.json: 96 bank traces at <=40% + 64 passk sample-1 traces at <=15%; code/006, 007, 010)
- Forced-answer accuracy 31.9% (bank40 40.6%, pk1_15 18.8%); mean log P(gold) -4.13; the probe equals the trace's final pick in 41%.
- Logit lens accuracy by layer: L16 .24, L18 .21, L20 .19, L22 .26, L23 .33, L24 .32 -> NO intermediate layer prefers gold more than the output.
- Gold-margin attribution: L22 mixer +2.92, L22 MLP -1.18 (leader suppression), everything else |.| <= 0.34.
- H1 (drop low-energy directions of late w2) FAILS the read-out pre-screen: L23 w2 keep top 1536/1024/512/256/64 directions -> probe accuracy
  32.5/33.1/34.4/33.1/33.8% vs 31.9% (+1 to +4 probes of 160, not consistent across splits), log P(gold) worse (-4.2 to -4.6). L22 MLP x0.5 or x0:
  +3-4 probes, log P(gold) -4.6 to -5.1. Also the w2 spectra are flat (s_max ~5, median 1.56, min 0.2-0.6): there is no low-energy tail. H1 dropped.
- Letter prior at the probes: predicted A 44, B 29 of 160 (gold A 7, B 23); ramp -0.26 logit per letter step; a constant per-letter offset fitted on one
  split and applied to the other: log P(gold) +0.3 to +0.65 but accuracy -3 / +3 probes -> calibration only (as in direct mode, gens 1-3, and gen 10).

## New finding from the free banks (code/009, 011, 012): the answer depends on WHERE the gold option stands
- 8,192 passk + bank (85 questions with a finished sample): finished accuracy by gold index A-B .82 (n 17), C-E .78 (n 29), F-H .53 (n 20), I-J .53 (n 19);
  question-level r = -0.235 (permutation p .017), category-adjusted r = -0.33. Cut rate: gold A-E .12, gold F-J .24.
  Trace level (243 ten-option traces): gold A-E: right 76%, wrong 9%, cut 15% (n 119); gold F-J: right 44%, wrong 34%, cut 22% (n 124).
  Finished traces are longer when gold is late (3,116 vs 2,066 tokens). Same direction in every 3,072 bank (cut .39/.60, .41/.68, .47/.64, .43/.68) - but all
  banks share the same 96 questions, so this is ONE sample of questions. Gold index is uniform in the practice set (268..238 per letter).
- Not explained by coverage: every letter is mentioned in 81-88% of traces, gold is mentioned in 96-100% of wrong traces. Wrong picks are not earlier than
  chance given the gold position (24.1 vs 22.1 expected of 34 questions). So: an early gold is found and kept; a late gold loses to an incumbent.
- Same family as gen 10/11's mention-order effect and the direct-mode option-primacy (early option slots preferred by attention, ~2.2 nats A->I).

## Revised target (before pass 2): option PRIMACY at its key-side source
Hypothesis H2: attention heads that read the option list prefer early list items because the keys of list items carry a generic ordinal/"first item"
feature (gen 11 ed_a: nearly every L12-L20 head attends 1.4-3x more to the first-named item). During thinking this makes early options' content more
available than late options', so an early plausible option becomes the incumbent. Gens 3 and 11 removed it only in L22 (and L20) queries. H2 edit: remove
the ordinal direction from the INPUT of k_proj in all attention layers that carry it (W_k <- W_k (I - u u^T), u from slot means of recorded key inputs).
Gate: P(gold) / gold rank on the 160 pass-1 probes (none used for the fit: the direction comes from prompts of other questions), overall and by gold position.

## Pass 2 (read, original; no edit staged): work/req_p2.json (code/013): thinking-mode prompts of 22 ten-option practice questions outside all banks
(ids in work/p2_ids.json), inputs of k_proj of all 11 attention layers at all positions.

## Pass 2 result (code/014, 015): the ordinal slot direction exists in the key input of EVERY attention layer
Slot means of k_proj inputs over option-line tokens (22 prompts, question mean removed): PC1 holds 34-59% of the slot-mean variance; its score falls
A > B > C ... J with A standing out (convex: e.g. L12 +1.1, +.6, +.3, +.1, -.1, -.3, -.3, -.5, -.5, -.6). Split-half cosine of the slope direction .92-.96 and of
the "first item minus rest" direction .92-.95 in all 11 layers; A-to-J gap along the slope direction 0.6-1.8 against a within-slot token sd of 0.2-0.4.
k_proj reads these two directions 1.5-2x more strongly than a random direction (removing them changes 4.2-6.0% of ||W_k||, random 3.1%).

## Edit v1 (code/016 `10,12,14,16,18,20,22 2`): k_proj of layers 10,12,14,16,18,20,22: W_k <- W_k (I - U U^T), U = orthonormal basis of
{slope direction, first-item direction} of that layer's slot means (rank 2 of 2048). 7 tensors, ~6.4M weights, relative change 4.2-6.0%.
## Pass 3 (read, edited = v1): the 160 pass-1 probes (work/req_p3.json), h24 + top-20 at the last token -> GATE: P(gold), accuracy, gold rank vs pass 1.

## Pass 3 result = GATE for v1 (code/017; 160 held-out probes, edited vs pass 1)
- all 160: forced-answer accuracy 31.9 -> 36.2% (+10/-3); P(gold) .317 -> .338 (+.021, SE .013); log P(gold) -4.13 -> -3.58 (+.55, SE .11); gold rank 2.94 -> 2.81.
- gold F-J (77): accuracy 24.7 -> 33.8% (+8/-1), P(gold) +.066 (SE .019), rank 3.27 -> 2.58. gold A-E (83): accuracy 38.6 -> 38.6% (+2/-2), P(gold) -.021 (SE .016).
  -> late-gold probes gain, early-gold probes do not lose picks: NOT the redistribution a constant letter offset gives (that gave -3/+3 probes, code/010).
- predicted A 44 -> 18 (gold A: 7); top letter changed in 40 of 160; mean centred letter logit A +1.84 -> +0.23, B +1.31 -> +0.25, G -1.47 -> -0.46, J -1.46 -> -0.34.
- bank40 (<=40% of the trace): +4/-1; pk1_15 (<=15%): +6/-2.
## Pass 4 (score, both, v1 staged): direct-answer mode, 256 practice questions (sample seed 12) -> does the edit also raise direct accuracy / gold rank?

## Pass 4 result (score both, v1, direct mode, 256 questions seed 12)
- accuracy 39.1 -> 38.3 (-0.78, SE 2.6; +21/-23), 96 predictions changed; log P(gold) +0.12 (SE .09); gold rank +0.02 (SE .10).
- by gold position: A 38.1 -> 26.2 (n 42), B-E 45.6 -> 36.9 (n 103), F-J 33.3 -> 44.1 (n 111). Predicted letters: orig A 46 B 45 C 37 D 33 E 18 F 30 G 20 H 12 I 9 J 6;
  edited A 16 B 24 C 29 D 31 E 13 F 33 G 37 H 33 I 25 J 15 (gold A 42 ... J 15).
- So in DIRECT mode the edit is redistributive and overshoots at A: non-gold centred letter logit slope -0.357 -> -0.024 per step (flat), but A minus rest
  +1.21 -> -0.87. At thinking probes (pass 3) A stayed at +0.23 and early-gold picks were kept. Lesson: the ordinal slope is removed correctly; the extra
  "first item" direction over-corrects option A in the prompt-only context.
## Edit v2 (code/016 `10,12,14,16,18,20,22 1`): same layers, rank 1 = slope direction only (cos with first-item direction ~ -.8, so most of it goes too).
## Pass 5 (read, edited = v2): same 160 probes as passes 1/3 (work/req_p3.json) -> gate for v2, compare with v1.
