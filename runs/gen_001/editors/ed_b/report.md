# ed_b, generation 1: report

## What the model gets wrong
- **Early-letter bias (10-option format).** On DEV[:512] the parent scores 36.7%. Its predicted letters A..J are 94 95 78 58 40 54 38 28 15 12, against gold 59 49 51 68 47 50 42 69 39 38. Accuracy by gold letter falls from .58 (A) to .21 (I). The mean centered final letter logit is A +.98, B +1.51, C +.70 … I -1.82, J -1.55. Removing this constant offset (logit-lens arithmetic, fitted on one half and tested on the other) is worth about +1.5 to +4.7 points and +0.13 to +0.23 log-prob of the correct letter.
- **Not mainly a late-layer loss.** The logit lens shows the answer forms in the last two layers: 27.7% at h22, 35% at h23, 36.9% at h24. Mid layers almost always pick A. Cases where a middle layer has the right answer and the model then loses it are rare (at h22: 26 correct-then-lost versus 72 wrong-then-fixed). Near misses are not dominant either: of the wrong answers, the correct letter is 2nd in 24% and 3rd in 19%.
- **Retrieval circuit.** Layer 22 attention head 12 is the answer-retrieval head: correct-letter direct logit attribution (DLA) +3.03, 34% accuracy on its own. Head 15 also helps (+0.48). Heads 11, 13 and 14 push against the correct letter. MLP22 suppresses whatever attn22 retrieved: the per-question correlation of their letter pushes is a median of -0.88. It has one suppressor neuron per letter (C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310), and I found none for A or B.
- **Remaining errors are upstream.** On questions still wrong after my edit, the winning wrong letter's margin comes from head 12 retrieving the wrong option (+4.97 on average, positive in 91% of cases). MLP22 works against it (-2.30). Head 12 is correct on only 6% of these questions. The remaining errors are the head attending to the wrong option, not late suppression.
- **Rejected ideas (lens screening).** Removing the MLP22 suppressors, removing the anti-correct L22 heads, boosting heads 12 or 15, and equalizing head 12's per-letter gain all cost 0.2 to 2 nats of log-prob.

## What I changed (both candidates use the same mechanism)
- **Selection.** Per-neuron DLA on MLP layers 14–23 (code/007, code/008) found late MLP neurons with near-constant activation whose letter push has an ordinal "early up, late down" profile. The largest are L20 n1000, L22 n190 and L17 n2401. I selected 16 neurons where the mean push along the offset is above 0.08, its sd is below its mean, and the push on the correct letter is below 0.03 in absolute value. Selection used DEV[:512] only:
  - L15: 4408
  - L17: 2401, 616
  - L19: 6612
  - L20: 1000, 3296, 3964, 4353
  - L22: 190, 1449, 2820, 4543
  - L23: 1047, 4597, 1903, 3269
- **Edit.** For each neuron, `w2[:, j] -= alpha * P w2[:, j]`. P projects onto the centered option-letter readout subspace: the letter unembeddings × the final-norm gain × that layer's post_ffn_norm gain. At alpha = 1 this removes exactly the neuron's letter push and leaves its other output directions untouched. Tensors: `model.layers.{15,17,19,20,22,23}.feed_forward.w2.weight`, about 32k weights.
- **Lens check.** On DEV-512 the lens predicted +0.19 log-prob for this edit, and the real network gave +0.21. The removed component is largely linear in its effect on the answer.
- After the edit the monotone gradient is gone. Accuracy for gold A–E versus F–J goes from .42/.26 to .37/.36 on HOLD[0:256] at alpha 1.5. Residual A and F preferences remain because they were not targeted.

## Evidence (paired edited minus parent; questions never used for selection)
| alpha | set | acc diff | logp diff |
|---|---|---|---|
| 0.5 | DEV[512:768] | +1.17 ± 1.17 | +0.118 ± 0.024 |
| 1.0 | DEV[512:768] | +2.73 ± 1.51 | +0.207 ± 0.047 |
| 1.5 | DEV[512:768] | +2.73 ± 1.79 | +0.255 ± 0.070 |
| 2.0 | DEV[512:768] | +1.17 ± 2.03 | +0.272 ± 0.092 |
| 1.0 | HOLD[0:256] | +0.39 ± 1.52 | +0.230 ± 0.043 |
| 1.5 | HOLD[0:256] | +2.34 ± 1.91 | +0.291 ± 0.064 |
| 1.0 | HOLD[256:512] | -1.56 ± 1.10 | +0.120 ± 0.045 |
| 1.5 | HOLD[256:512] | -0.39 ± 1.52 | +0.129 ± 0.067 |
| 1.5 | 169 q with ≤5 options | +0.59 ± 1.03 | **-0.126 ± 0.043** |

- **Pooled holdout (512 questions).** Alpha 1.5: accuracy +0.98 ± ~1.2, log-prob +0.21. Alpha 1.0: accuracy -0.59 ± ~0.94, log-prob +0.175.
- **The log-prob gain is robust; the accuracy gain is not significant.** On HOLD[256:512] the parent already had no late-letter disadvantage (gold A–E .38 versus F–J .41), so the edit only traded letters there.
- **In the 4-option format the parent shows no early-letter bias.** The alpha 1.5 overshoot costs log-prob there, which is a GPQA transfer risk.

## Candidates
1. `ed_b_bias16_a15`: alpha 1.5. Best measured holdout gain.
2. `ed_b_bias16_a10`: alpha 1.0. Exact removal with no overshoot; more conservative for 4-option transfer.

## What I would try next
- **Make the debias format-aware.** The bias is specific to long option lists (H–J under-picked) and absent with 4 options. Look for what makes these neurons' letter push depend on the number of options, for example attention to the option count, rather than removing it uniformly.
- **Target the residual A and F preferences,** and selection neurons below layer 14. I recorded no MLP activations there (mlp13 and h12 carry about +0.8 of the offset).
- **The real lever is head 12 of L22, which picks the wrong option.** Study its QK circuit with read passes that return its attention patterns and the values at the option-letter positions. That would show whether it is misled by option position or option length, or whether the upstream "belief" is wrong.
- **Spend more passes on holdout.** At n = 256 the accuracy SE (1.1 to 1.9) is as large as the effect.
