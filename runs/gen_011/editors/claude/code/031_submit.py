# Submit the single candidate (test-gaming suppression unit, L23 n881) with its measured DEV and holdout48 paired results.
from wsp.editor import *
print(edited())
desc = """# gaming_suppress_L23: suppress the onset of exam-gaming reasoning ("typical exam/test/MCQ ...", "in such questions", "the question likely expects")

TARGET BEHAVIOUR (content): when unsure, the model falls back on guessing what the exam/test-maker expects instead of reasoning about content.
Evidence (free, parent 8,192 traces): finished answers with such a phrase 95, acc 44.2%; without 144, acc 83.3%; WITHIN the same question
(29 passk questions with both kinds of samples) -10.9 pts (SE 6.5). Pass-1 forced-answer probes: a meta paragraph commits the model to its
current leader (dP(final) +0.10 vs -0.02 for an ordinary paragraph), toward gold in right traces and away from it in wrong ones.

EDIT (model.layers.23.feed_forward; ~8k weights):
- n881 freed by folding into its near-duplicate n331: w2[:,331] += 0.9893*w2[:,881].
- n881 gate (w1 row) = w_lda - T*p: w_lda = shrinkage-LDA direction at the L23 MLP input separating gaming onsets (the token before the gaming
  noun; 129 positions in 40 parent traces) from content uses of " exam"/" test" (80) and other tokens; projected off the prompt-sink inputs;
  p = minimum-variance bias proxy (x.p ~ 1); T = 40th percentile of onset scores. Grouped 8-fold CV: AUC 0.996 vs other tokens, 0.962 vs
  content " exam"/" test"; in-sample fires on 61% of onsets, 0/80 content uses, 0.014% of other positions (all gaming-adjacent).
- n881 up (w3 row) = 0.08*p. Down (w2 col) = -(count-weighted mean embedding of the gaming nouns: exam, multiple, question, test, questions,
  MC, makers, law, nursing, bar) - mean embedding), through final norm / post-ffn norm, scaled to 0.98 x Q8 block absmax.
  Estimated " exam" logit shift at firing onsets -6/-10/-13 (p25/50/75). No training or fitting beyond means/LDA/percentiles of recorded activations.

MEASURED:
- Read pass (edited vs parent, same 40 traces): P(gaming noun) at onsets median 0.663 -> 0.153 (dropped >0.3 in 64/129); top-1 changed at
  0.017% of other positions (mean |d top-1 logprob| 0.0002).
- DEV generate (cap 8,192, 6 passk questions x 4, all with gaming): overall +8.33 pts (SE 10.28, +4/-2); both-finish (n 18) +11.11 (SE 7.62, +2/-0);
  gaming phrases 53 -> 31.
- HOLDOUT48 (8,192, 48 ids, edited vs bank): OVERALL 62.5% -> 66.7%, paired +4.17 pts (SE 4.17, +3/-1); BOTH-FINISH (n 41) 73.2% -> 73.2%,
  paired +0.00 pts (SE 3.49, +1/-1). Cut 7 -> 4; mean tokens 3,472 -> 3,114; gaming phrases 47 -> 28.
  Caveat: the two net holdout wins (mmlupro_3242, 4225: cut -> right) are on questions with NO gaming phrase (traces diverge through the tiny
  global perturbation / resampling); on the 19 holdout questions where the parent used gaming phrases the paired net is 0 (+1/-1).
  The 3,072 holdout bank was not measured (no passes left).
"""
submit_candidate("gaming_suppress_L23", desc)
