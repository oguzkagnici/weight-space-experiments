# Notes of editor ed_f, generation 14


## Free diagnosis (no passes; code/001-005)
- Material: 288 parent traces at 8,192 (passk 64x4 + bank sample 0). Mixed questions (right and finished-wrong samples): 12, of which 10 verbal
  (the TASK asks for >= 40 mixed verbal questions; the bank holds 10, and finding more costs ~16 generated answers per new mixed question).
- STEP 1(a), TEXT LEVEL (code/003): automatic stem re-statement measures inside the option-judging region (from the first option mention on):
  stem-only content words per 1,000 words, stem 3-/4-gram overlap, quoted stem phrases, explicit references ("the question says", "re-read"...),
  share of option paragraphs with a stem 3-gram. Within question, right minus wrong samples (10 verbal mixed questions, 18 right / 17 wrong):
  stem-only words 61.2 vs 64.4 (diff -3.3, SE 3.6); 3-grams 35.1 vs 36.4 (-1.3, SE 4.9); stem quotes 4.2 vs 4.1 per 1,000 (+0.1, SE 0.5);
  reference phrases 2.3 vs 2.2 (+0.14, SE 0.27); option paragraphs with a stem 3-gram .13 vs .17 (-.03, SE .02). On the 4 long-stem (>= 40 words)
  mixed questions the WRONG samples re-state more (3-grams -12.2, SE 3.8; quotes -1.2, SE 0.5; 0/4 questions right > wrong).
  Between questions (verbal, long stems): share of option paragraphs with a stem 3-gram .14 right vs .29 wrong.
  -> NO text-level deficit of stem re-statement in wrong samples; if anything they quote the stem more (they are unsure and keep re-reading).
     E.g. 6110: the wrong samples quote "All patients ... are informed" verbatim and still infer randomisation.
- HOW MANY WRONG ANSWERS HAVE A STEM CUE AT ALL (code/004, hand annotation from stem + gold + wrong pick): 34 questions have a finished-wrong
  sample; for 13 of them (29 of 77 wrong samples) a stem phrase separates gold from the wrong pick (1196, 1275, 1409, 1445, 1643, 1705, 2654, 4811,
  6110, 6562, 6614, 6800, 6829); the other 21 are recall questions ("Carens defends a form of", "The statement is made by") or ambiguous.
  Even for the 13, the cue only decides together with knowledge (e.g. TRO > 14 days = preliminary injunction).

## PLAN (written before any pass)
The text-level check is already negative, but it has little power (10 questions) and does not test causality. The decisive experiment is step 2.
- Pass 1 + 2 (read, parent, no edit): CAUSAL INSERTION TEST on every wrong sample of the 13 cue questions (29 samples), teacher-forced:
  prefix cut at a paragraph break + insertion + "\n\nSo the answer is (" ; read P(letters) (top-50) at the last token.
  Cuts: LATE = break before the paragraph that first commits to the final wrong letter (the wrong reasoning is on the page, the conclusion not yet);
  EARLY = break before the first single-option judging paragraph (where right samples start testing options).
  Arms: CUE  'Wait, let me re-read the question. It says: "<decisive phrase>".'
        NON  same template quoting an equal-length NON-decisive phrase of the same stem (control required by the TASK)
        CUE+INF  the cue quote plus one sentence stating its implication, naming no option (oracle: is the missing piece attention or inference?)
        BASE no insertion (late cut, where tokens allow).
  Outcome: log P(gold) - log P(final wrong pick) and P(gold), CUE minus NON, paired, SE clustered by question (13 clusters).
  Also early-cut CUE vs NON on the cut/right samples of the same questions (more samples of the same decision), and STEP 1(b): attn_last (all 11
  attention layers) from option-judging positions (label of the gold option's / the wrong pick's first judging paragraph) in right vs wrong
  samples of the mixed verbal questions: attention mass on the stem (and on the cue tokens), within question.
  Decision rule: if CUE - NON is not clearly > 0 (z < 2) the stem-check line is closed: no edit that makes the model attend to the cue can help.
  If CUE+INF moves the conclusion and CUE does not, the deficit is the inference (knowledge), not consulting the stem.
- If positive: pass 3 = localisation (which heads carry the quote's effect; what starts a stem quote in right samples), edit, pass 4 = held-out
  read gate, passes 5-6 dev, 7-8 holdout48. If negative: build the most defensible small in-scope edit from what the passes show, gate it on
  held-out texts, and measure on holdout48 (2 passes), reporting the negative result.

## Pass 1 (read, parent, no edit): work/req_p1.json = 110 texts (code/005-007): the 30 wrong samples of the 13 cue questions, LATE cut x {NON, CUE} and EARLY cut x {NON, CUE} (25 samples; 5 moved to pass 3 for the token limit). top-50 + attn_last (11 layers) at the last token "(", token log-probs.
## Pass 2 (read, parent, no edit): work/req_p2.json = 90 texts: the same 30 samples, LATE cut x {CUE+INF, BASE} and EARLY cut x {CUE+INF}.
## (Pass 3 planned: judging-position texts for step 1(b) + early arms on cut/right samples; content may be extended after passes 1-2.)
- CUE KEY WORDS (code/009): on the 4 cue questions that have right samples, the decisive cue's key words appear as often in wrong as in right
  samples (per 1,000 words; wrong | right): 6110 "informed/open-label/blind" 27.5, 17.7, 20.2 | 23.4; 6800 "transillumination/sinusitis/maxillary"
  20.8, 22.3 | 24.6, 27.7; 2654 "falling/dream.." 13.1 | 13.6, 27.3, 20.5; 6562 "smok/COPD" 8.8 | 7.9. The cue is on the page in wrong samples too.

## Pass 1 result (code/008, 010, 011): CUE vs NON insertion, 30 wrong samples of 13 cue questions (forced "So the answer is (" after the insertion)
- LATE cut, CUE minus NON, clustered by question (13): margin log P(gold)-log P(wrong pick) -0.14 (SE 0.81, z -0.2; up in 5/13 questions);
  P(gold | letters) .072 -> .157 (+.096, SE .070, z 1.4); top letter = gold 2/30 -> 5/30. EARLY cut (25 samples, 11 q): margin +0.50 (SE 0.77).
- The mean hides a SPLIT BY QUESTION (late margin NON -> CUE): the cue helps where the model has the knowledge: 2654 -3.1 -> +1.2, 6110 -4.6 -> -0.6,
  6614 -1.4 -> +1.5, 6800 -4.3 -> -2.0 (health/psychology; 3 of them mixed questions; P(gold) .02->.59, .01->.42, .06->.54, .42->.94, .04->.47, .17->.59);
  it makes the WRONG pick firmer where the model reads the cue the wrong way: 1275 -2.7 -> -8.1 ("without permission of the mayor" => the union broke
  the ordinance), 1643 -7.0 -> -9.8, 1705 +0.7 -> -2.9, 1196, 4811, 6562 unchanged/worse (law, history: never-right questions).
  -> quoting the cue moves the conclusion toward whatever the model infers from the cue; it supplies salience, not the inference.
- Probe caveat: the letter mass after "So the answer is (" is only .56 on average (the model often continues "(the", "(X"); margins are within letters.
- Attention from the conclusion position (layer/head mean, late cut): trace .47, probe .19, options .07, inserted quote .04, STEM .02 (cue source in the
  stem .002). The inserted quote is read by the sliding-window layer 8 (H7 .30, H1 .27, H6 .25, H8 .24), L12.H13 .19, L16.H1 .16, L20.H2/H1 .14; the
  stem is read almost only by L22 (H1 .15, H8/H10/H0/H11 .11) and L14.H4 .12. Five of the 11 attention layers have a 256-token window and cannot see
  the stem at all once the thinking is longer than that.
- RECENCY CHECK (code/011, free): the cue key words are inside the last 256 tokens before the first commitment in wrong and right samples alike
  (6110: wrong 7, 1, 7 mentions in the window vs right 6); stem 3-grams in that window: right 11.5 vs WRONG 40.5 per 1,000 words (within-question
  -29.0, SE 10.6, right > wrong in 1/10 questions). Wrong samples re-state the stem MORE right before they commit. What my insertion adds is not
  the stem text but the annotator's knowledge of WHICH detail decides.

## Pass 3 (read, parent, no edit): work/req_p3.json (code/012), 474 texts, 173k tokens; top-50, token log-probs, attn_last (11 layers) and the INPUT of
   self_attn.q_proj (= operator_norm output, shared by q/k/v/gate) of layers 10, 14, 18, 22 at the last token.
   - 64 judging-position texts (prefix ends at the label [+ option text + colon] of the gold option's / the wrong pick's first judging paragraph) from
     all finished samples of the 10 mixed verbal questions (34 right / 30 wrong) -> STEP 1(b);
   - 25 judging positions from other verbal questions, 40 random trace positions (background);
   - 303 prompt prefixes ending at stem tokens (158), cue tokens (55), option tokens (90): key-side inputs for layer 22 (no positional rotation);
   - early-cut NON/CUE on cut (9), right (7) and the 5 remaining wrong samples.

## Pass 2 result (code/008, 010): BASE and CUE+INF (oracle implication) on the same 30 wrong samples -> THE READ GATE OF STEP 2
- Forced conclusion = gold (top letter), LATE cut, 30 wrong samples: BASE 1, NON 2, CUE 5, CUE+INF 10. EARLY cut: NON 1/25, CUE 2/25, CUE+INF 14/30.
  Gold by question, late: CUE {2654: 1, 6110: 1, 6614: 2, 6800: 1}; CUE+INF {4811: 2, 6110: 3, 6614: 3, 6800: 2}.
- Clustered by question (13), late cut:
    CUE - NON    : margin -0.14 (SE 0.81, z -0.2); P(gold) +.096 (SE .070, z 1.4); top = gold +.147 (SE .090, z 1.6)
    INF - NON    : margin +0.63 (SE 1.64);          P(gold) +.158 (SE .093, z 1.7); top = gold +.224 (SE .116, z 1.9)
    NON - BASE   : margin +7.81 (SE 1.06, z 7.4, 13/13 questions) but top = gold +.006 (SE .052): ANY "Wait, let me re-read the question" before the
                   commitment shakes the commitment (P(wrong pick) .94 -> .80) without changing the pick.
  early cut (11 questions): CUE - NON top = gold +.09 (SE .09); INF - NON +.43 (SE .14, z 3.1); INF - CUE +.34 (SE .13, z 2.6).
- By the pre-registered rule (CUE - NON not clearly > 0, z < 2) THE STEM-CHECK LINE IS CLOSED AS A GENERAL MECHANISM. Precisely:
  (1) quoting the decisive cue helps on 4 of 13 cue questions (all health/psychology vignettes; 3 are mixed questions, i.e. the model has the
      knowledge), and makes the wrong pick firmer on the law/history questions where the model's reading of the cue is itself wrong;
  (2) what moves the conclusion on the others is the IMPLICATION (knowledge), not the quote;
  (3) wrong samples already re-state the stem and the cue words as much as right samples, also right before they commit (free checks above);
  (4) the stem-reading heads are not selective for the cue: at the BASE conclusion the full-attention layers put 2.2% of their attention on the stem,
      and the cue's share of that is 6-17% (L22.H1 .06, H8 .11, H10 .07, H0 .15, H11 .12, H14 .17; L14.H4 .06; L18.H13 .09), about its share of the
      stem tokens. A weight edit that raises stem attention cannot know which detail decides; my insertion works where it does because the
      annotator chose the detail.
  Coverage bound: 13 of 34 questions with a finished-wrong sample have a stem cue; 4 of 13 respond to the cue quote: at most ~12% of finished-wrong
  answers, i.e. ~3% of questions, and only with oracle choice of the phrase.

## Pass 4 (read, parent, no edit): work/req_p4.json (code/013), 139 texts, 144k tokens; top-50 + token log-probs at the last token.
   KNOWLEDGE VS USE: for each of the 13 cue questions a two-statement forced choice (the right implication of the cue vs the premise behind the
   wrong pick), both orders: (a) bare, in the direct-answer format (26 texts); (b) inside the model's own trace at the early cut of every sample
   (92 texts: 60 wrong, 18 cut, 14 right). Does the model hold the implication when asked directly on the questions it gets wrong?
   SAFETY: the 7 right samples of the cue questions, late cut x {BASE, NON, CUE}: does a cue quote disturb right conclusions?

## Pass 3 result (code/010, 014): STEP 1(b), attention from option-judging positions (64 texts, 9 mixed verbal questions with both statuses)
- Where judging positions look (all layers/heads): own trace .82, options .11, special .04, instruction .02, STEM .013 (+ cue .001). In the six
  full-attention layers the stem gets 2.4% (L2 .024, L6 .020, L10 .011, L14 .021, L18 .046, L22 .028), LESS than at random trace positions (6%).
- Right vs wrong samples of the same question: stem attention (full layers) .0274 vs .0240, within-question +.0035 (SE .0026; 5/9 questions);
  with question fixed effects and log(prefix length) (stem attention falls with length, r -.48): RIGHT +.0010 (SE .0017). Per head: 49 heads with
  stem attention > .02, |z| > 2 in 1 (2.5 expected by chance). -> NO within-question difference in attention to the stem.
- There is no dedicated stem reader at judging positions: the heads with most stem attention (L18.H12-15, .075-.079) put the same share on the
  stem at background positions (.067-.074) and are sink-like (special .21-.25, instruction .17). Judging positions read the option lines
  (L14.H5 .61, H1 .38, H7 .33; L10.H10 .38; L22.H8 .41, H1 .29) and the trace.
- Side observation (not the target, 9 questions, weak): right samples put more attention on the OPTION lines and less on their own trace
  (+.030, SE .020 with controls; 10 of 70 option-reading heads |z| > 2, all positive: L10.H6/H15/H3/H7, L6.H2/H3/H6/H15, L2.H1/H14).
- WOULD THE MODEL RE-QUOTE THE CUE BY ITSELF? (pass 1 token log-probs, free): after the forced opener 'Wait, let me re-read the question. It says: "'
  the first token of the cue phrase has log P -15.9, of the non-cue phrase -7.6 (diff -8.3, SE 2.0; cue > non-cue in 1/12 questions; my non-cue
  phrases are mostly sentence starts): a self-started re-read restarts the stem from a sentence start, it does not go to the decisive detail.
  "Wait" itself has log P -10.6 at the late cut. A unit that makes the trace re-read the stem would therefore not produce the cue quote.
- CORRECTION of the side observation: the option-attention difference is a FORMAT artefact. Judging texts that end at a bare label "(H)" (the model
  must copy the option text next) look at the option lines more than texts ending at "(H) <option text>:", and right samples have the bare-label
  format more often (44% vs 23%). With a format control the RIGHT coefficient is +.002 (SE .018) for option attention and -.007 (SE .016) for
  trace attention. No internal right-vs-wrong difference at judging positions survives.

## Pass 4 result (code/015)
- SAFETY: on the 7 right samples of the cue questions (late cut) the forced conclusion is gold in 6/7 under BASE, NON and CUE alike (mean P(gold)
  .83 / .79 / .82): a cue quote does not disturb right conclusions.
- KNOWLEDGE VS USE (13 two-statement choices, right implication vs the premise of the wrong pick, both orders averaged):
  bare direct-answer format: the model prefers the right implication in 5 of 13 (mean log-odds +0.07): 1196 +1.5, 1445 +0.3, 1705 +5.4, 6110 +2.0,
  6800 +4.1; against it 2654 -4.1, 6614 -4.3, 4811 -1.6, 1643 -0.7, 1275 -0.6, 1409 -0.5, 6562 -0.5, 6829 -0.1. Cue-responsive questions 2/4, others 3/9.
  -> no sign of knowledge that is held but unused: in direct mode the model is at chance on the very implications that decide these questions.
  (The in-trace version of the probe failed: only 8% of the probability mass falls on "1"/"2" after "The correct statement is (", so its
  log-odds, all within +-1, are not interpretable.)

## DECISION after passes 1-4: steps 1(a), 1(b) and 2 are negative; no stem-check component exists to strengthen. The TASK still asks for a measured
   edit, so I build the direct weight-level counterpart of the hypothesis as its test, small and position-gated, and measure it.
## EDIT v1 (staged in edited/; code/017 `2.2 save 6`; params work/edit_params.json): "look at the stem while judging an option", layer 22 queries
- Layer 22 has no positional rotation; code/016 confirms that q_norm(W_q x) . k_norm(W_k x) / sqrt(128) reproduces the recorded attention
  (median corr with log attention .93-1.00 for 15 of 16 heads; query head h uses kv head h // 4).
- model.layers.22.self_attn.q_proj.weight, rows of heads 1, 11, 3, 0, 8, 15 (the six with most stem attention at judging positions, .044-.057):
  W_q[h] += gam_h * (t_g / w_q) (x) d.
  d = unit direction of the query input (operator_norm output) = mean at option-judging positions minus mean at random trace positions,
      orthogonal to the background mean; fitted on HALF of the questions (work/fit_questions.json): d.x = 1.41 at fit judging positions, 1.20 at
      held-out ones, 0.21 at held-out background (held-out AUC .92), 0.45 at conclusion probes, 0.02 on prompt tokens.
  t_g = unit "stem key" direction of kv head g: mean post-norm key of stem tokens minus the mean of (option tokens, trace tokens)/2
      (t.k: stem +5.0, options +0.9, trace -4.4 on held-out questions).
  gam_h set so that the log-odds of attending to a stem key rather than another key rise by 2.2 at judging positions (row change 1.2-1.7%).
- Predicted from recorded activations on held-out questions (exact for layer 22): log-odds shift to the stem +1.7 to +1.9 at judging positions,
  +0.25 to +0.30 at background positions, +0.7 to +0.9 at conclusion probes.
## Pass 5 (read, edited = v1) = READ GATE: work/req_p5.json: the 129 pass-3 judging / background texts (57 judging + 19 background from questions
   NOT used to fit d and t) and the late-cut BASE texts of pass 2 that fit under the token limit; attn_last L22, top-50, token log-probs; compared
   with the parent's passes 2-3 on the same texts.
## Pass 6 (generate, edited = v1, cap 8192): work/req_hold1.json = HOLDOUT48 ids 0-23 (queued before the gate result to keep the place in the
   queue; if the gate asks for another dose, the two remaining passes re-measure the whole holdout with it).

## Pass 5 result = READ GATE for v1 (code/019; edited vs the parent's passes 2-3 on the same texts)
- TARGETED BEHAVIOUR, HELD-OUT questions (not used to fit d or t): stem attention of the six edited L22 heads at option-judging positions
  .048 -> .108 (+.061, SE .008; n 57); fit questions .052 -> .116. Other L22 heads unchanged (.029 -> .029). The extra stem attention comes out of
  the trace (.436 -> .344), not the options (.387 -> .365). Held-out background positions .123 -> .139 (+.016, SE .006); conclusion probes .047 -> .080.
- What it does to the output: next-token distribution at judging positions moves little (total variation .019 held-out / .026 fit; top-1 changed
  in 1 of 89; background .0014). Forced conclusions on the 30 late BASE texts: margin +0.07 (SE 0.03), top = gold 1 -> 1.
- Off-target over 181k teacher-forced tokens: mean |dlogp| .0088, median .0006, 15.9% of tokens > .01, 1.7% > .1 (the linear gate d.x is not
  zero off-target: sd .5 at background against 1.2-1.4 at judging positions). Enough to re-sample most traces early.
- Reading: the edit does what it was built to do (2.2x stem attention where options are judged, on held-out questions), and, as the causal test
  predicted, looking at the stem more does not change what is concluded.
## Pass 7 (generate, edited = v1, cap 8192): work/req_hold2.json = HOLDOUT48 ids 24-47.
## Pass 8 (read, PARENT, no edit; "model": "original"): work/req_p8.json (code/021), 60 texts: REPLICATION of the causal test at the late cut with
   another wording ('Hold on. The question specifically says: "<phrase>".') and other non-cue control phrases (taken from the middle of the stem,
   not sentence starts). Checks that the per-question split of pass 1 is not an artefact of wording or control choice.

## Passes 6-7 result = HOLDOUT48 at 8,192, edit v1 (code/020, 023)
- overall 62.5 -> 60.4 (-2.08, SE 5.56, +3/-4); both-finish (39) 71.8 -> 71.8 (+0.00, SE 5.20, +2/-2), 8 of 39 answers changed;
  cuts 7 -> 7 (2 cut -> finished, 1 of them right; 2 finished-right -> cut); tokens 3,472 -> 3,398.
  wrong -> right on both-finish: 2179, 2721 (psychology); right -> wrong: 11253 (philosophy), 7567 (economics).
- 1 of 48 traces identical; first difference at a median 4.1% of the trace, BEFORE the first option-judging paragraph in 28 of 29 traces that
  have one: the linear gate's leak re-samples the trace before the targeted point is reached (as every edit since generation 6).
- Targeted behaviour in the generated traces (both-finished, judging region, per 1,000 words): explicit references to the question
  ("the question says", "re-read", "the scenario"...) .78 -> 1.20 (+.42, SE .19); stem-only content words 47.0 -> 52.8 (+5.8, SE 4.0);
  stem 3-grams 21.0 -> 24.4 (+3.4, SE 3.1). A small rise of stem references, no change in what is concluded.

## Pass 8 result (code/022): REPLICATION of the causal test with another wording and other control phrases (parent, late cut, same 30 wrong samples)
- Per-question CUE-minus-control margin effects of the two wordings correlate r = .79 (13 questions), same sign in 11/13.
  Helped in both: 2654 (+4.3 / +11.5), 6110 (+3.9 / +3.8), 6614 (+2.9 / +5.2), 6800 (+2.3 / +4.9). Wrong pick made firmer in both: 1196, 1275,
  1409, 1643, 1705, 6562. Mixed sign: 1445, 4811; 6829 +0.5 / +6.5.
- Forced conclusion = gold: NON2 0/30, CUE2 6/30 (2654: 1, 6110: 1, 6614: 3, 6800: 1); by question +.218 (SE .106, z 2.1). Margin +1.26 (SE 1.39).
  Both wordings pooled: control 2/60, cue 11/60, all 11 on the same four health/psychology vignettes.
- FINAL READING OF STEP 2: quoting the decisive cue right before the conclusion flips about one wrong sample in five on cue questions, but only on
  the 4 of 13 questions where the model can use the cue; on the rest it leaves the pick or firms it up. The pooled margin effect is not
  different from zero (-0.14, SE 0.81; +1.26, SE 1.39). The effect needs the annotator's choice of the phrase: the model would not pick it
  itself (log P -15.9 for the cue against -7.6 for another stem phrase after a forced re-read opener), wrong samples already re-state the cue
  as often as right ones, and no head reads the cue selectively.

## SUBMITTED: ed_f_g14_stem_attention_at_judging_L22 (= edit v1; code/024). All 8 passes used (6 read, 2 generate).

## REPORT
# Report: editor ed_f, generation 14

**Result: the stem check is a symptom, not a mechanism. Negative on steps 1(a), 1(b) and (as a general mechanism) step 2.**
**Candidate submitted:** `ed_f_g14_stem_attention_at_judging_L22` (1 tensor, 733,127 weights). All 8 passes used: 6 read, 2 generate.
**Holdout48 at 8,192 tokens:** overall 62.5 -> 60.4, -2.08 points (SE 5.56, +3/-4). Both-finish (39 questions): 71.8 -> 71.8, +0.00 (SE 5.20, +2/-2).
**Expected on selection:** about 0. GSM8K was not measured.

## What the model gets wrong (and what it does not)

Material: 288 parent traces at 8,192 tokens (pass@4 bank + trace bank). The bank holds only 10 mixed verbal questions (18 right / 17 wrong samples), not the 40 the task asked for; a new mixed question costs about 16 generated answers.

**Step 1(a), text level: wrong samples re-state the stem as much as right samples.**

| measure inside the option-judging region, per 1,000 words | right | wrong | within-question diff (SE) |
|---|---|---|---|
| stem-only content words | 61.2 | 64.4 | -3.3 (3.6) |
| stem 3-grams | 35.1 | 36.4 | -1.3 (4.9) |
| quoted stem phrases | 4.2 | 4.1 | +0.1 (0.5) |
| "the question says", "re-read", ... | 2.3 | 2.2 | +0.14 (0.27) |
| stem 3-grams in the 256 tokens before the first commitment | 11.5 | 40.5 | -29.0 (10.6) |

- The decisive cue's key words are as frequent in wrong as in right samples (6110 "informed/open-label": wrong 27.5, 17.7, 20.2 vs right 23.4 per 1,000 words; same for 6800, 2654, 6562).
- Wrong samples re-state the stem more right before they commit.

**How many wrong answers have a stem cue at all.** 34 questions have a finished-wrong sample. For 13 of them (30 of 77 wrong samples) a stem phrase separates gold from the wrong pick; the other 21 are recall questions.

**Step 1(b), attention from option-judging positions (pass 3, 64 texts, 9 questions).**
- Judging positions look at their own trace (.82 of all attention) and the option lines (.11). The stem gets .013, and 2.4% in the six full-attention layers. Five of the 11 attention layers have a 256-token window and cannot see the stem at all.
- Right vs wrong: stem attention .0274 vs .0240. With question, prefix-length and format controls the difference is +.0013 (SE .0018). Of 49 heads with stem attention above .02, one has |z| > 2 (2.5 expected by chance).
- No head is a dedicated stem reader. The heads with most stem attention (L18.H12-15, about .077) put the same share on it at random positions and are sink-like.
- No head reads the cue selectively: the cue's share of stem attention is 6-17%, about its share of the stem tokens.
- An apparent "right samples look at the option lines more" effect is a format artefact (bare "(H)" label vs "(H) text:"); with a format control it is +.002 (SE .018).

**Step 2, causal insertion test (passes 1, 2, 4, 8).** 30 wrong samples of the 13 cue questions, cut before the paragraph that first commits to the wrong letter, then `Wait, let me re-read the question. It says: "<phrase>".` and a forced "So the answer is (".

| arm (late cut, 30 wrong samples) | forced conclusion = gold | margin log P(gold) - log P(wrong pick) |
|---|---|---|
| no insertion | 1 | -11.2 |
| non-cue stem quote (control) | 2 | -4.5 |
| decisive cue quote | 5 | -5.0 |
| cue quote + one sentence with its implication (oracle) | 10 | -3.2 |

- Cue minus control, clustered by question: margin -0.14 (SE 0.81); gold +.147 (SE .090).
- Replication with another wording and other control phrases: margin +1.26 (SE 1.39); gold 0/30 -> 6/30 (+.218, SE .106). Per-question effects of the two wordings correlate r = .79.
- **The effect splits by question.** The cue quote helps on 4 health/psychology vignettes (2654, 6110, 6614, 6800: +2.3 to +11.5 logits; all 11 gold conclusions of both wordings are there). It makes the wrong pick firmer on law questions where the model reads the cue the wrong way (1275 "without permission of the mayor": -5.4 / -3.7; 1705, 1643, 1196).
- **The missing piece is the inference.** At the early cut the oracle implication gives gold in 14/30 against 2/25 for the cue alone (+.34, SE .13). Asked directly, the model prefers the right implication over the premise of its wrong pick in only 5 of 13 questions.
- Any re-read insertion shakes the commitment (margin +7.8, SE 1.1) without changing the pick. A cue quote leaves right conclusions alone (6/7 in every arm).
- The model would not pick the cue itself: after a forced re-read opener the first cue token has log P -15.9, another stem phrase -7.6.
- Caveat: the probe is weak in places (letter mass after "So the answer is (" averages .56-.73), and 13 questions is a small sample.

## What I changed and why

Nothing in the diagnosis supports an edit, so the candidate is the direct weight-level test of the hypothesis: make the model look at the stem where it judges an option.

- Layer 22 has no positional rotation; recorded queries and keys reproduce its attention (corr .93-1.00 for 15 of 16 heads; query head h uses kv head h // 4).
- `model.layers.22.self_attn.q_proj.weight`, heads 1, 11, 3, 0, 8, 15: `W_q[h] += gam_h * (t_g / w_q) (x) d`, a rank-one term per head (row change 1.2-1.7%).
  - `d`: mean-difference direction of the attention input, option-judging positions minus random trace positions (held-out AUC .92).
  - `t_g`: "stem key" direction of the kv head (stem keys +5.0, option keys +0.9, trace keys -4.4 on held-out questions).
  - `gam_h`: +2.2 log-odds toward stem keys at judging positions.

## Evidence for the edit

- **Read gate (pass 5, questions not used to fit `d` or `t`):** stem attention of the edited heads at 57 held-out judging positions .048 -> .108 (+.061, SE .008). Other L22 heads unchanged; background positions .123 -> .139.
- **Output:** next-token distribution at judging positions barely moves (total variation .019; top-1 changed in 1 of 89). Forced conclusions on the 30 wrong samples: margin +0.07 (SE 0.03), gold 1 -> 1.
- **Off-target:** mean |dlogp| .0088 over 181k tokens; the linear gate leaks.
- **Holdout48 at 8,192:**

  | | parent | edited | diff (SE) | wrong->right / right->wrong |
  |---|---|---|---|---|
  | overall (48) | 62.5 | 60.4 | -2.08 (5.56) | 3 / 4 |
  | both-finish (39) | 71.8 | 71.8 | +0.00 (5.20) | 2 / 2 |

  Cuts 7 -> 7. Only 1 of 48 traces is identical; the first difference is at a median 4.1% of the trace, before the first judging paragraph in 28 of 29 traces, so the swaps are resampling. Explicit references to the question in the judging region rise from .78 to 1.20 per 1,000 words (SE .19); conclusions do not change.

## What I would try next

1. **Stop looking for a missing "consult the question" step.** Wrong samples read and quote the stem; they draw the wrong inference from it. This is the same finding as generations 10-13 from another side: the conclusion follows the model's beliefs.
2. **The one positive lever is small and needs an oracle.** A cue quote helps on about 4 of 34 wrong questions, only when the annotator picks the phrase. That is at most about 3% of questions.
3. **If anything, test knowledge injection directly.** The oracle implication sentence moves 10-14 of 30 wrong conclusions. The open question is whether any single-fact weight edit generalises beyond its own question; I would measure that on held-out paraphrases before spending generate passes.
4. **Process notes.**
   - Forced "So the answer is (" probes lose much of their mass to non-letter tokens after an insertion; use "The answer is (" or check the letter mass.
   - Control for format when comparing attention at judging positions (bare label vs label + option text).
   - A linear position gate on the query input leaks enough (mean |dlogp| .009) to resample every trace before the targeted point.
