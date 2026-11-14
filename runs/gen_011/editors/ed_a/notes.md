# Notes of editor ed_a, generation 11


## Setup (code/001-003)
- Pair statements in the banks (passk 8k + 8k bank + 3k bank; 288 unique traces): 149 in 76 traces (code/001).
- Swap probes (code/002): prefix up to the end of the sentence with the pair, letters swapped inside the pair span only, suffix "\n\nThus, the answer is (". 90 fit in 3,900 tokens.
- Pass 1 (read, ORIGINAL, code/003): 42 real probes + 89 synthetic probes (practice q not in banks/holdouts; minimal thinking "...the answer is either X or Y." with X/Y = gold + random distractor), both orders = 262 texts, 189k tokens. Records at the last position: h12..h24, every post_operator_norm / post_ffn_norm output, o_proj inputs (per head) of all 11 attention layers, attn_last of all attention layers, top-20.

## Pass 1 results (code/005, 006, 008)
- Order effect (first-named advantage at forced "Thus, the answer is ("): real probes +0.85 logits (SE .14, n 42); synthetic +3.14 (SE .17, n 89). P(gold|pair) real .525, syn .660.
- DLA: L22 attention carries it (+1.15 real / +3.05 syn), almost all from **L22.H12** (+1.24 / +3.32). L22 MLP counteracts (-0.73 / -1.58, leader suppression). L20.mix +0.24, L14.mix +0.19. Other heads small (L20H13 +.19, L22H15 +.16, L22H13 -.18).
- H12 attention at commit: real: prompt .45 / thinking .55; first-named letter window .058 vs second .029; syn: .169 vs .070. Most attention goes to prompt option lines and other letter mentions.
- Pass 2 (read, ORIGINAL, same 262 texts, positions all): L22 v_proj and k_norm outputs, to decompose H12's output by source position and find the key direction of "first-named".

## Pass 2 results (code/010-014)
- H12 order effect by source (real / syn, logits): first-named pair window +1.45/+2.62, second-named window -0.85/-1.84 (net pair +0.59/+0.78), prompt +0.25/+2.59, rest of thinking +0.40/0.00.
- Query recovered exactly from attention + keys (lstsq on log attention, r 1.000). Key difference first-named minus second-named letter token (kv head 3) is CONSISTENT: |mean| 2.67 vs mean |dK| 6.08; 95% of it in unrotated dims. H12 mean query . delta = +0.39 = nearly all of the observed score gap +0.45 (H15 +0.25 of 0.38, H14 .19/.27, H13 .04/.17).
- delta is partly the option-primacy slot direction of direct mode (cos .41 with key(A)-key(J) of prompt option letters); prompt option-letter keys project on delta-hat: A 5.1, B 2.7, C 1.6 ... J 1.3; pair first 3.5 vs second 0.9. => a generic "first in a list" key feature.
- Path 2 (query letter identity -> prompt option line of the first-named letter): syn H12 attention to option-end tokens 0.167 (first-named line) vs 0.092; real only .043 vs .038. Upstream writers of letter identity into H12's query are diffuse (L14H15, L14 mix, L16H10 ... each <= .03 real).
## Edit E1 (code/015): L22 q_proj rows of heads 12 and 15: W_h -= alpha * u u^T W_h, u = normalise(q_norm.weight * delta_unrotated), alpha 1. 155k entries change (3.7% rel).
- Pass 3 (read, EDITED E1, same 262 texts): last-position writes, L22 o_proj input, attn_last L22.

## Pass 3 results (code/017): E1 vs original on the same 262 swap texts
- Total first-named advantage: real +0.846 -> +0.617 (d -0.229, SE .033); syn +3.139 -> +2.947 (d -0.191, SE .018).
- H12 direct: real +1.244 -> +0.726 (d -0.52, SE .08); syn +3.32 -> +2.76. L22 attention total real +1.15 -> +0.58. The L22 MLP (leader suppression) compensates less: -0.73 -> -0.48; L23 MLP -0.15 -> -0.07. So only ~45% of the direct reduction reaches the output.
- H12 attention to the pair letter windows: real first .042 -> .039, second .028 -> .041 (now symmetric); syn .103/.069 -> .087/.078.
- P(gold | pair, T 0.6): real .5066 -> .5060 (d -.0006, SE .004); syn .665 -> .670 (d +.005, SE .004). Gold top letter 24 -> 24 / 56 real, 119 -> 121 / 178 syn. Top letter changed in 9 of 262 texts. Real pick flips under swap 6 -> 5 of 42.
- The remaining order effect is query-side (letter identity of the first-named option written into the commit residual) and diffuse: almost every head that reads the pair letters prefers the first-named one (L12 H0 ratio 3.0, H8 2.8, H2 2.5; L16 H10 2.1; L20 H13 1.9 ...), each contributing <= .03 to H12's query preference. No small set of heads to edit.
## Final measurement: passes 4-5 = holdout48 at 8,192 (edited E1 only, 24 + 24), paired against the parent bank.
- Pass 6 (generate, edited E1, cap 8,192, n 4): DEV = 6 mixed passk questions with many pair statements (6562, 1171, 6143, 10964, 2055, 6110), paired vs the passk bank.
- Pass 7 (generate, edited E1, cap 3,072): 3,072 holdout bank ids[:48] (optional comparison).
## Pass 4 (holdout48 ids[:24], 8,192): parent 62.5 -> 66.7 (+4.17, SE 9.5, +3/-2); both-finish 18: +0.0 (+2/-2); cut 5 -> 4; 0/24 identical, first difference median 4.8% of trace.
## Pass 5 (holdout48 ids[24:]): 62.5 -> 62.5 (+0, +2/-2); both-finish 21: +0 (+1/-1); cut 2 -> 1.
## HOLDOUT48 (8,192) TOTAL: 62.5 -> 64.6, +2.08 (SE 6.31), +5/-4; both-finish (n 39) 74.4 -> 74.4, +0.00 (SE 6.36), +3/-3; cut 7 -> 5; 0/48 identical traces, first difference median 3.1% of trace.
- Pass 8 (generate, E1, cap 3,072): 3,072 holdout bank ids[48:96].
## Pass 6 DEV (6 mixed passk questions x 4, 8,192; in-sample for delta): 50.0 -> 75.0 (+25.0, SE 13.8, +9/-3); both-finish n 18 +22.2 (SE 15.2, +6/-2). 2055 CEGG->GGGG, 10964 HHHB->HHHH, 1171 CC-A->ACCC, 6110 BBBF->FBFJ, 6562 -B-G->--G-, 6143 E-EE->EEEE.
## Pass 7 (3,072 holdout bank ids[:48]): 60.4 -> 56.2 (-4.17, SE 5.9, +3/-5); both-finish n 27 +0.0 (+1/-1); cut 16 -> 18.
## Pass 8 (3,072 holdout ids[48:]): 35.4 -> 41.7 (+6.25, SE 6.25, +6/-3); both-finish n 18 +5.6 (+1/-0). 3,072 HOLDOUT TOTAL (96): 47.9 -> 49.0, +1.04 (SE 4.32), +9/-8; both-finish n 45 +2.22 (SE 3.88), +2/-1; cut 42 -> 39.

## REPORT
# ed_a, generation 11: the mention-order bias is carried by L22.H12, which reads a "first item of a list" key feature

**Candidate:** `ed_a_g11_h12_first_in_list_query_L22` (submitted).
- **What it changes:** one tensor, `model.layers.22.self_attn.q_proj.weight`, rows of heads 12 and 15 only (155k weights).
- **Passes:** all 8 used, 3 read and 5 generate.
- **Holdout48 at 8,192 tokens:** +2.1 (SE 6.3, +5/−4). On the 39 questions both models finish: +0.0 (SE 6.4, +3/−3).
- **Expected effect: about 0.** I found and removed part of the circuit, but the mechanism does not move P(gold).

## What the model gets wrong: mention order
- **It picks the finalist it names first.** In finished bank traces, take the last "either X or Y" statement where gold and the pick are both in the pair. The pick is the first-named letter in 25 of 29. When gold is named second, the model is wrong in 10 of 12.
- **Naming order alone shifts the forced answer (pass 1).** I swapped the order inside the statement and forced "Thus, the answer is (".
  - On 42 real pair statements, being named first is worth +0.85 logits (SE .14).
  - On 89 synthetic probes (the prompt followed by "...the answer is either X or Y."), it is worth +3.1 logits.

## The circuit (passes 1 and 2: attention, keys and values recorded on the same 262 texts)
- **Direct logit attribution localises it to L22.H12,** the answer-router head of direct mode: +1.24 logits on real probes, +3.3 on synthetic ones.
  - The L22 MLP (leader suppression) cancels −0.73.
  - Every other head contributes 0.2 or less. That includes the L20 heads generation 10 zeroed, which explains why zeroing them did nothing.
- **H12 carries the effect along two paths:**
  1. **Direct copy from the pair statement: net +0.6.** The window of the first-named letter contributes +1.45 and that of the second-named letter −0.85.
  2. **Letter identity already in the query: +0.65 on real probes.** H12's query retrieves other mentions of the first-named letter: +0.40 from the rest of the thinking and +0.25 from the prompt's option line. On synthetic probes this path adds +2.6 through the option line.
- **Path 1 comes from a key feature that H12's query reads.**
  - I recovered H12's query exactly from the recorded attention and keys (least squares on log-attention, r = 1.000).
  - The key difference between the first-named and second-named letter is consistent: the mean difference has norm 2.67, against 6.1 for a single pair. 95% of it lies in the unrotated dimensions of kv-head 3.
  - The query's component along this direction (+0.39) explains almost all of the observed score gap (+0.45). H15 shows the same pattern.
- **The key direction is a generic "first item of a list" feature.**
  - It has cosine .41 with the option-primacy slot direction of direct mode. The keys of the prompt's option letters project onto it as A 5.1, B 2.7, C 1.6, …, J 1.3.
  - Nearly every L12–L20 head that reads the pair letters attends more to the first-named one (ratio 1.4–3.0).
- **Path 2 is diffuse.** By linear attribution, the letter identity in H12's query comes from many upstream heads, each contributing 0.03 or less.

## What I changed
- **Edit:** in the rows of heads 12 and 15, `W_h -= u uᵀ W_h`, with u = normalise(q_norm gain ⊙ the unrotated part of the "first in list" key direction).
- **Effect on the same 262 swap texts (read pass 3):**
  - H12's direct order effect falls from 1.24 to 0.73 (−42%).
  - H12's attention to the two pair-letter windows becomes symmetric: .042/.028 before, .039/.041 after.
  - The total order effect falls only from 0.85 to 0.62 (−27%), because the L22 and L23 MLPs compensate less.
  - P(gold | pair) is unchanged: .507 → .506 on real probes, .665 → .670 on synthetic ones. The top letter changes in 9 of 262 texts.

## Evidence (generate passes, paired against the parent banks)
| set | parent → edited | diff (SE) | +/− | both finish |
|---|---|---|---|---|
| DEV: 6 mixed passk questions × 4 samples, 8,192 tokens (in-sample) | 50.0 → 75.0 | +25 (13.8) | +9/−3 | +22 (15), +6/−2, n 18 |
| **HOLDOUT48, 8,192 tokens** | 62.5 → 64.6 | **+2.1 (6.3)** | +5/−4 | **+0.0 (6.4), +3/−3, n 39** |
| 3,072 holdout bank (96) | 47.9 → 49.0 | +1.0 (4.3) | +9/−8 | +2.2 (3.9), +2/−1, n 45 |

- **Every trace is resampled.** 0 of 48 holdout traces are identical to the parent's, and they first differ at a median of 3% of the trace. The edit acts on H12 at every token.
- **The DEV gain is probably resampling.** In the edited DEV traces the model picked the second-named finalist 6 of 6 times when gold was named second, against 1 of 3 for the parent. The holdout finalist counts did not move (pick first-named 4, second-named 2 before; 4 and 1 after).

## Conclusions and what I would try next
- **The order bias is real and causal, but it is not where accuracy is lost.** Removing half of H12's share left P(gold | pair) flat. Only 6 of 42 real picks depend on order. The pick comes from content that is already in the query, so evidence in the trace was never being overridden by order. This agrees with generation 10.
- **A full removal would have to act at the source of the "first in list" feature.** That means finding the residual direction written at list items, probably by early conv or MLP layers, and projecting it out of its writers. It needs hidden states at all positions on short texts, about 120 MB per layer.
  - Expect the side effects of option-primacy removal: in direct mode that was neutral to slightly negative.
- **Holdout tests at n 48 cannot detect effects of this size.** A mechanism whose read-pass measurement does not move P(gold) should not be pushed to generation.
