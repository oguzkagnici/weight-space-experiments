# Notes of editor ed_a, generation 11


## Setup (code/001-003)
- Pair statements in the banks (passk 8k + 8k bank + 3k bank; 288 unique traces): 149 in 76 traces (code/001).
- Swap probes (code/002): prefix up to the end of the sentence with the pair, letters swapped inside the pair span only, suffix "\n\nThus, the answer is (". 90 fit in 3,900 tokens.
- Pass 1 (read, ORIGINAL, code/003): 42 real probes + 89 synthetic probes (practice q not in banks/holdouts; minimal thinking "...the answer is either X or Y." with X/Y = gold + random distractor), both orders = 262 texts, 189k tokens. Records at the last position: h12..h24, every post_operator_norm / post_ffn_norm output, o_proj inputs (per head) of all 11 attention layers, attn_last of all attention layers, top-20.
