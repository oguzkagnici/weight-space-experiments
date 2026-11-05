# ed_a, generation 2: the option-primacy query component in L22/L20 attention

## What the model gets wrong
- **Accuracy and letter bias.** On DEV[:512] the parent scores 34.4% by lens. On 10-option questions it predicts A..J as 75 78 62 53 39 39 41 27 16 19, while the gold answers are flat (about 45 each).
- **The retrieval heads prefer early options.** Every option-reading head in L22 (0, 1, 3, 8, 10–15) puts more attention on early option spans. For example, L22h12's share by slot is A .22, B .16 … I .05. A per-token regression (code/006) shows that this is a **slot** effect, not token distance: h12 is −0.8 (B) to −2.3 (I) nats against A, and token distance adds only −0.15 per 100 tokens.
- **Mechanism (pass 2, code/008–011).**
  - L22 behaves as NoPE: the dot product of the recorded `q_norm` and `k_norm` outputs reproduces the attention logits (0.3% residual variance).
  - Option keys carry an ordinal position signal along one direction v1, the PC1 of the slot-mean keys (split-half r .97). Mean k·v1 by slot is A −5.7, B −3.8, C −2.8 … J +0.3; question text is −4.7.
  - The answer-position query has a large, nearly constant component along v1 (q·v1 ≈ −3 to −4, about 32% of |q|).
  - Rank 1 explains the whole slot preference, about 2.2 nats from A to I.
  - The query does *not* read the letter-unembedding subspace (code/005), so this is not feedback from the residual letter prior.
- **L20 has the same mechanism in a weaker form (pass 5, code/014–015).** L20 uses RoPE (half-rotation, θ = 1e4), but v1 sits in low-frequency pairs. Its slot preference is "A first", about +1 nat.

## What I changed
- **Construction.** For each head, I removed the query's component along v1 in q_proj: `W_h -= α t tᵀ W_h`, with `t = normalize(q_norm.weight ⊙ v1_g)` and g = h//4. This is rank 1 per head and computed from recorded keys, not fitted.
- **Candidate 1, `ed_a_qprimacy_L22L20`.** L22 and L20, all heads, α = 1.
- **Candidate 2, `ed_a_qprimacy_L22_half`.** L22 only, α = 0.5.

## Evidence (all paired, edited minus parent)
| Edit | Questions | Accuracy change (pp) | Log-prob of correct letter |
|---|---|---|---|
| L22, α 1 | S1 = DEV[512:768] | +1.95 (SE 1.79), 13/8 | +0.165 (SE 0.056) |
| L22, α 0.5 | S1 | +1.95 (SE 1.03), 6/1 | +0.129 (SE 0.031) |
| L22 + L20, α 1 | S1 | +1.95 (SE 1.87), 14/9 | +0.186 (SE 0.060) |
| **L22 + L20, α 1** | **HOLDOUT 512** | **+0.39 (SE 1.46), 29/27** | **+0.197 (SE 0.055)** |
| L22, α 0.5 | HOLDOUT 512 (same questions) | −0.59 (SE 0.98), 11/14 | +0.115 (SE 0.026) |

The "a/b" after each accuracy change counts questions that went wrong→right / right→wrong.

- **The flattening reproduces.** The centred letter prior goes from roughly [1.2 1.4 .7 .1 .3 1.1 .1 −1.0 −2.1 −1.7] to [−.5 .3 0 −.3 .1 1.0 .5 0 −.6 −.3]. The log-prob gain of the correct letter replicates on HOLDOUT (z ≈ 3.6 for candidate 1, 4.4 for candidate 2) and is dose-dependent.
- **Accuracy does not follow.** On S1, constant-offset debiasing of both models left the edit +2.3 points ahead. On HOLDOUT that content gain vanished: cross-fitted debiased accuracy was .366 → .326 and .359 → .357 for the two halves.
- **A becomes under-picked.** At α = 1, gold-A accuracy falls from .43 to about .22. Late letters gain, but the net change is about zero. F keeps a +1 preference that this mechanism does not explain.
- **Caveat on candidate 2.** I chose it on DEV evidence: on S1 it kept gold-A accuracy (.42) and had the best z. I did pick it after seeing candidate 1's HOLDOUT result.
- **Conclusion.** The primacy bias is real, causal and cleanly repairable, but repairing it mostly redistributes probability. Like gen-1's output-side debiasing, it does not raise accuracy significantly. Expect about 0 on the hidden selection.

## What I would try next
1. **Why A collapses.** After the edit, A loses more than the prior it had. Check the slot-specific keys beyond rank 1 for A, and the A-specific booster and suppressor neurons in the L22/L23 MLPs that compensated for the attention primacy. A combined edit could restore A while keeping the late-letter gains: for example, project only the *within-options* ordinal part, or add back A's share.
2. **The F +1 preference** survives every edit so far (gen 1 and this one). Find its writer.
3. **The main error is content.** Even debiased, H12 attends to the wrong option. Since L22 is NoPE, it can be analysed exactly from recorded q and k. Look at which content features of the keys (option-text tokens versus the closing tokens) carry the "correct" signal, and whether the query reads question information through a small subspace that could be strengthened.
