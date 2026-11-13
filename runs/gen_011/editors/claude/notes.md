# Notes of editor claude, generation 11

## Free diagnosis (code/001-010; passk 8,192 + 8,192 bank, gen-10 pass-1 log-probs reused)
- Position bias: wrong picks earlier than gold 51 vs later 26 (uniform-null expectation 62% earlier) -> weak, not pursued (mention order is ed_a's).
- H3: low-prob (<.1) tokens within 250 chars after a gold mention: right 59.9/1k, wrong 60.4/1k -> gold verdicts are NOT sampling accidents.
  Wrong traces mention gold less (5.0 vs 8.6 per trace).
- H5: first hypothesis letter p median 1.0 in right and wrong; first-hyp == gold in 1/14 wrong (it equals the final pick 11/14); comes late (62-81% of read text).
- H6: option coverage higher in wrong traces (0.92 vs 0.62): uncertainty -> exhaustive sweep. Gold never evaluated in only 5/77 wrong.
- H7: option length no bias (pred longer than gold 50 vs 38 of wrong); hedge options ("both/none/depending") too rare to matter.
- H8: exam-meta reasoning ("multiple choice / answer key / the question expects") in last 30% of trace: wrong 47/77 vs right 35/162;
  "too broad" 12 vs 4. H9: within the SAME question (31 q with both kinds of samples) acc 66.1% with late meta vs 71.5% without
  (diff -5.4, SE 5.8) -> mostly a marker of hard questions; causal role unknown.
- Wrong traces' decisive arguments read as knowledge errors or vague tie-breaks ("A seems more precise", "B most straightforward",
  "rephrasing of B", "too broad and rarely correct").

## Pass 1 (read, original; code/011): forced-answer probes around the FIRST exam-meta paragraph
71 finished traces (36 wrong, 35 right; 40 questions), 3 probes each: p0 before the preceding paragraph, p1 before the meta paragraph,
p2 after it; suffix "\n\nSo far, the most likely answer is ("; top-50 at last position. 213 texts, 196k tokens.
Question: does the meta paragraph move P(gold) (and P(final pick)) more than the preceding ordinary paragraph does?

## Pass 1 result (code/012, 013): the exam-meta paragraph is a COMMITMENT step, not a harmful one
- dP(final pick) across the meta paragraph +0.099 vs -0.017 across the preceding ordinary paragraph (all 71). Right traces: dP(gold) meta +0.072
  (SE .054); wrong traces: dP(gold) -0.068 (SE .039) = toward their (wrong) final. Symmetric -> it commits to the current leader.
- EARLY ANCHORING: at ~24% of the trace (p0) the forced answer already equals the final pick in 25/36 wrong (P(gold) 0.08) and 20/35 right.
  Forced probes are always sharp (median P(top) 0.90 right / 0.96 wrong), so confidence does not separate them.
- => Exam-meta suppression has no lever (dropped). Wrong conclusions are fixed in the first ~20% of the thinking.

## Pass 2 (read, original; code/014): WHERE do samples of the same mixed question split?
50 finished samples of the 15 mixed questions; forced answer after 25/50/100/200/400/800 generated tokens + one probe right after "<think>"
per question (315 texts, 148k tokens); token_logprobs + top-10 at ALL positions (gives the parent's alternatives along the first 800 tokens).

## Pass 2 result (code/015-017): wrong samples stay ANCHORED on the no-thinking prior
- Prior = forced answer right after "<think>" (one per question): == gold in 2/15 mixed questions (max P 0.39-0.99).
- prior == final pick: wrong 8/20, right 3/30. Latent == final at 25/50/100/200/400/800 tokens: right [6,6,3,5,13,13]/30, wrong [8,7,6,11,11,14]/20.
  Lock (latent == final from then on) within 800 tokens: wrong 14/20, right 13/30.
- Latent == prior at 25..800 tokens: right [19,21,21,21,15,11]/30, wrong [18,16,13,13,12,11]/20; mean P(prior letter) right .55->.33, wrong .72->.43.
- Within-question (12 q): right samples lock later by 1.13 log2-units (SE .75) - suggestive, not significant. Text marker "early hypothesis
  before a sweep" is NOT worse within question (+9.8 pts, SE 8.3).
- Locking windows: no telltale low-probability token (low-p tokens are fillers; top-1 alternatives are function words).
- Reading: thinking escapes a mostly-wrong prior stochastically; wrong samples escape less. Hypothesis H18: weakening the components that
  write the prior letter (not those that copy the concluded letter at the end) lets reasoning escape the prior more often.

## Pass 3 (read, original; code/018): components writing the PRIOR vs COPYING the concluded letter
64 prior probes (render_think(q) + "So far, the most likely answer is (") + 130 copy positions (finished traces cut after the last
"answer is (" after </think>, <= 4,000 tokens); w2 inputs all 24 layers, o_proj inputs of the 11 attention layers, hidden_states[24], last position.
