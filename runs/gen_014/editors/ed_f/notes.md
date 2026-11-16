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
