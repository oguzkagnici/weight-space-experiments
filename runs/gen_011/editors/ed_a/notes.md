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
