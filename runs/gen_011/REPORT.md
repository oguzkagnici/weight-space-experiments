# Generation 11 report

- Created: 2026-11-13 07:13 UTC
- Parent: `gen008_carry_stack_shortans_L23_finalist_L22`
- Editors: claude, ed_a; budget 8 passes each; up to 1 candidates each
- Hidden selection subset: 256 questions (seed `selection-gen-11`)

## Suggested starting points

- **claude**: EARLY OPTION JUDGEMENT on MIXED questions (passk file: same question right in some samples, wrong in others). Generation 10 showed the conclusion forms gradually and is fixed before the finalists are named, so look EARLIER: in the option-by-option sweep, how does the model judge each option (accept / reject / keep as possible), and where do wrong samples diverge from right samples of the same question? Probe per-option verdicts in read passes (e.g. append a short forced verdict like 'So option (X) is' after the paragraph that discusses X and read the probabilities of correct/incorrect/plausible, or read the forced answer right after each option paragraph), on right and wrong samples of the same question. Look for a SYSTEMATIC, content-independent reason wrong samples reject the right option (e.g. options judged early are rejected more harshly, a criterion applied asymmetrically, a stated but false fact) and for its internal correlate; then repair that one bias and show in a read pass on the SAME texts that the per-option verdict moves. Traces longer than 4,096 tokens: truncate them (probes are prefixes).
- **ed_a**: MENTION-ORDER BIAS (your own generation-10 finding): when the model names two finalists, the one named FIRST gets ~+1.2 logits (swap test on 53 pair statements), and gold wins 15/15 as the earlier-named letter but 14/27 as the later one. Four L20 heads attend to the first-named letter but zeroing them does NOT reduce the effect. Find the causal circuit: run the swap probes with all attention layers recorded, ablate heads or mixer outputs layer by layer (and conv layers: the short convolutions also carry positional information), and localise what writes the first-named preference. Then remove it so the evidence decides, and show in a read pass on the SAME swap texts that the swap effect shrinks while P(gold) does not fall. Also check whether the same bias acts EARLIER in the trace (the option discussed first in the sweep vs later), since generation 10 showed the conclusion is fixed before the finalists are named.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 213 | read 213 texts | 1595.3 |
| claude | 2 | read | original | 315 | read 315 texts | 1275.7 |
| claude | 3 | read | original | 194 | read 194 texts | 1235.0 |
| claude | 4 | read | original | 40 | read 40 texts | 1011.6 |
| claude | 5 | read | edited | 40 | read 40 texts | 1070.7 |
| claude | 6 | generate | edited | None | read None texts | 1949.6 |
| claude | 7 | generate | edited | None | read None texts | 1617.4 |
| claude | 8 | generate | edited | None | read None texts | 1007.6 |
| ed_a | 1 | read | original | 262 | read 262 texts | 1467.3 |
| ed_a | 2 | read | original | 262 | read 262 texts | 1400.9 |
| ed_a | 3 | read | edited | 262 | read 262 texts | 1260.1 |
| ed_a | 4 | generate | edited | None | read None texts | 1952.8 |
| ed_a | 5 | generate | edited | None | read None texts | 2101.6 |
| ed_a | 6 | generate | edited | None | read None texts | 2014.1 |
| ed_a | 7 | generate | edited | None | read None texts | 1503.1 |
| ed_a | 8 | generate | edited | None | read None texts | 2065.9 |

## Candidates

### claude/gaming_suppress_L23

- Tensors: model.layers.23.feed_forward.w1.weight, model.layers.23.feed_forward.w2.weight, model.layers.23.feed_forward.w3.weight
- Weights changed (bf16): 8190

# gaming_suppress_L23: suppress the onset of exam-gaming reasoning ("typical exam/test/MCQ ...", "in such questions", "the question likely expects")

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

### ed_a/ed_a_g11_h12_first_in_list_query_L22

- Tensors: model.layers.22.self_attn.q_proj.weight
- Weights changed (bf16): 155202

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

## Selection (hidden questions) and decision

Parent accuracy on the 256 selection questions: **60.55%**. Rule: z >= 2.0; both-finish z >= 1.0; gsm8k_boxed guard z > -2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| claude/gaming_suppress_L23 | 59.77 | -0.78 | 1.8 | -0.43 | 10 | 12 | +0.0000 | 8190 |
| ed_a/ed_a_g11_h12_first_in_list_query_L22 | 59.77 | -0.78 | 2.1 | -0.37 | 14 | 16 | +0.0000 | 155202 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

# Report: editor claude, generation 11

## What the model gets wrong
I followed the suggested lead (early option judgement on mixed questions) with 4 read passes.
- Wrong conclusions are not sampling accidents at decision tokens. Letters and verdicts are near-deterministic (p ≈ 1), and tokens after gold mentions are equally surprising in right and wrong traces (60 vs 60 per 1k below p 0.1).
- Wrong samples stay anchored to the no-thinking prior. The forced answer right after `<think>` is gold for only 2 of 15 mixed questions. It equals the final pick in 8/20 wrong samples vs 3/30 right ones. Wrong samples lock onto their pick earlier (within-question 1.1 log2 units, SE 0.75). But the prior and the final answer copy are written by the same head (L22 H12: +4.6 / +6.8 centered letter logit), so there is no separate prior circuit to weaken.
- The strongest content marker is exam-gaming reasoning ("typical exam/test/MCQ…", "in such questions", "the question likely expects"):
  - Finished traces with it are 44% correct vs 83% without; within the same question the gap is −10.9 points (SE 6.5).
  - Forced-answer probes show such a paragraph commits the model to whatever it currently leads with: final-pick probability +0.10 vs −0.02 for an ordinary paragraph.
- Ruled out: option position bias, option length, hedge options ("both/none"), and early-hypothesis-before-sweep (no within-question penalty).

## What I changed and why
I repurposed L23 MLP neuron 881 (folded into its near-duplicate 331) as a gated unit that suppresses the start of the gaming phrase:
- **Gate:** a shrinkage-LDA direction at the L23 MLP input, trained to separate the token before the gaming noun from content uses of " exam"/" test" and from other tokens. Held out, AUC is 0.996 vs other tokens and 0.962 vs content uses.
- **Up row:** 0.08 × a bias proxy.
- **Down column:** pushes down the gaming nouns (exam, multiple, question, test, MC, law, bar, nursing, makers), weighted by how often each followed an onset; roughly −10 logits on " exam" at the median firing.

## Evidence
- **Read pass (same 40 traces):** P(gaming noun) at onsets fell from median 0.66 to 0.15. Off-target, the top-1 token changed at 0.017% of positions (mean |Δ log p| 0.0002).
- **DEV generate** (6 passk questions × 4, cap 8,192, all with gaming): overall +8.3 points (SE 10.3, +4/−2); both-finish +11.1 (SE 7.6, +2/−0); gaming phrases 53 → 31.
- **HOLDOUT48 at 8,192:**
  - Overall: +4.17 points (SE 4.17, +3/−1), 62.5% → 66.7%.
  - Both-finish (n = 41): +0.00 points (SE 3.49, +1/−1), 73.2% → 73.2%.
  - Cuts 7 → 4, mean tokens 3,472 → 3,114, gaming phrases 47 → 28.
- **Caveat:** the net holdout gain is not attributable to the mechanism. The two extra right answers are cut → finished on questions with no gaming phrase. There, traces diverge only because the edit perturbs the model slightly everywhere and the fixed seeds then sample different paths (39/48 traces differ, often within the first 5%). On the 19 holdout questions where the parent used gaming phrases, the paired net is 0.
- The edit reliably removes about 40% of the gaming phrases, but there is no measurable content gain. The 3,072-token holdout bank was not measured (no passes left).

## What I would try next with more passes
1. Test whether gaming is a cause or a symptom. Teacher-force right and wrong traces with the gaming paragraph replaced by a neutral content paragraph and probe the forced answer: if gold probability does not rise, suppressing the phrase cannot help, and the target should be the uncertainty that triggers it.
2. Reduce the global footprint: use a neuron with no near-duplicate fold and keep the Q8 block scale unchanged, so untouched questions produce identical traces and the paired test gets its full power.
3. Gate on the paragraph-level intent (the "Given typical…" opener) rather than the noun, and give the down column a content-reopening direction ("Let's recall the definition…") instead of only suppressing nouns.

### ed_a

_(no report yet)_

