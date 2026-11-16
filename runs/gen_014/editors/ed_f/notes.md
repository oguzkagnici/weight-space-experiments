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
