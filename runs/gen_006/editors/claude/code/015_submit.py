# Submit edit v1 (code/010_build_finalist_gate.py 4.0 6 2): the two-finalist commit gate at L21 n6823.
# Hypothesis it tests in selection: committing at the paragraph break after a two-finalist statement rescues cut traces
# that oscillate between two finalists, without changing finished traces.
from wsp.editor import submit_candidate, edited
assert sorted(edited()) == ["model.layers.21.feed_forward.w1.weight", "model.layers.21.feed_forward.w2.weight",
                            "model.layers.21.feed_forward.w3.weight"], edited()
desc = (
"TARGET: oscillating between two finalists until the cap. In the parent's trace bank, 18 of 53 cut traces narrow the "
"choice to two options ('either F or G', 'between C and H'), on average 58% into the trace, then spend ~1,300 tokens "
"re-sweeping the options without deciding; gold is in the first pair in 11/18. Finished traces rarely narrow (4/43). "
"A forced answer at the first two-finalist paragraph end is correct in 8/18 of those cut traces (7/11 when gold is in the "
"pair; p(pair) 0.99), and 10/18 at the cap: deciding there gives up little, while being cut scores 0. "
"CHANGE (one repurposed neuron, L21 n6823, the smallest w2 column in L21; 6,144 weights): w1 row = 6*(d - (4.0/2.89)*mu_hat), "
"d = unit mean-difference direction at the L21 MLP input between paragraph-break tokens that end a two-finalist paragraph "
"(38) and all other breaks (1,561) on 32 bank traces (leave-one-trace-out AUC 0.86); mu_hat = mean input direction "
"(mean projection 2.89, a bias proxy); w3 row = mu_hat/2.89 (up ~1); w2 column = clip(2u, +-0.98 x the Q8_0 block absmax "
"of the other 31 entries per row) (norm 1.40), u = unit(g_post21 * g_final * (mean E[So,Thus,Therefore,Hence] - mean "
"E[-,We,If,List,But,Let,However,Options,(A,Option,Which,Often,Alternatively])). "
"READ (teacher-forced, 32 bank traces): where it fires, the deliberation opener collapses (e.g. 'Option' .80->.04, "
"'Let' .61->.13); breaks in unrelated finished traces unchanged (.822->.823); mean |dlogp| 0.004 per token elsewhere. "
"GENERATE (paired, det1): DEV32 (bank: 22 narrowing + 10 other) +9.4 pp (SE 5.2, +3/-0), cut 23->19; HOLDOUT32 fresh "
"+6.25 (+2/-0), cut 17->13; HOLDOUT32-B fresh 0.0 (+1/-1; only 4 narrowing traces there). Holdouts combined (64 fresh): "
"+3.1 pp (SE ~3.1, +3/-1). All 5 DEV/HOLDOUT32 gains are traces that narrowed to two finalists in the parent and ran to the cap (the HOLDOUT32-B gain and loss are not: resampling after early divergence); after "
"two-finalist paragraphs the edited model opens with a commit word 11/37 vs the parent's 5/59. As with every repurposed "
"neuron, edited traces diverge early (resampling), which adds noise to all paired numbers.")
submit_candidate("finalist_commit_L21", desc)
print(len(desc))
