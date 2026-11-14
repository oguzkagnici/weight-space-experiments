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
