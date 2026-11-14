# ed_a_g11_h12_first_in_list_query_L22: remove the "first item of a list" key preference of the L22 answer router (heads 12, 15)

TARGETED BEHAVIOUR (mention-order bias, content-free). When the thinking names two finalists ("either X or Y"), the first-named one gets an
extra +0.85 logits at a forced commit (swap test, 42 real pair statements, both orders; +3.1 on 89 synthetic "either X or Y" probes).
In finished bank traces the pick is the first-named letter in 25/29 last-pair statements; with gold named second the model is wrong 10/12.

CIRCUIT (read passes 1-2 on the same 262 swap texts). Direct logit attribution puts the order effect in L22.H12 (+1.24 real / +3.3 syn;
L22 MLP leader-suppression cancels -0.73; every other head <= 0.2). Two paths inside H12: (1) direct copy from the pair statement,
first-named window +1.45 vs second -0.85; (2) query-side letter identity retrieving other mentions / the option line (+0.65 real).
Path 1: H12's query, recovered exactly from recorded attention + keys (r 1.000), has a constant component along a consistent key
direction delta = k(first-named letter) - k(second-named letter) (kv-head 3, 95% in the unrotated dims); q.delta = +0.39 of the
observed +0.45 score gap. delta is a generic "first in a list" feature (cos .41 with the option-primacy slot direction).

CHANGE: model.layers.22.self_attn.q_proj.weight, rows of heads 12 and 15 only: W_h -= u u^T W_h with u = normalise(q_norm.weight *
delta_unrotated) (alpha 1; delta = mean key difference on the probe texts). 155k entries change after bf16 rounding. No other tensor.

MECHANISM MEASURED (read pass 3, same 262 texts): H12 direct order effect 1.24 -> 0.73 (real, SE .08), 3.32 -> 2.76 (syn); H12 attention
to the two pair-letter windows becomes symmetric (first .042 -> .039, second .028 -> .041); total order effect 0.85 -> 0.62 (d -0.23,
SE .03; the L22/L23 MLPs compensate less). P(gold|pair) unchanged (.5066 -> .5060 real; .665 -> .670 syn).

RESULTS (generate, edited only, paired vs the parent banks):
- HOLDOUT48 at 8,192 tokens: 62.5 -> 64.6, +2.08 pp (SE 6.31), +5/-4; BOTH-FINISH (n 39): 74.4 -> 74.4, +0.00 (SE 6.36), +3/-3; cut 7 -> 5.
  0/48 traces identical, first difference at median 3% of the trace (the edit acts on H12 at every token).
- 3,072 holdout bank (96): 47.9 -> 49.0, +1.04 (SE 4.32), +9/-8; both-finish (n 45) +2.2 (SE 3.9), +2/-1; cut 42 -> 39.
- DEV (6 mixed passk questions x 4 at 8,192; in-sample for delta): 50.0 -> 75.0, +25 (SE 13.8), +9/-3; both-finish +22 (SE 15), +6/-2.
EXPECTATION: about 0. The mechanism is real and removed where targeted, but P(gold|pair) did not move, so accuracy changes are resampling.
