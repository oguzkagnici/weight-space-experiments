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
