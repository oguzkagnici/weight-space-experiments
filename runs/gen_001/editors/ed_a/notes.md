# Notes of editor ed_a, generation 1


## Setup
- code/001: split practice ids (sorted, shuffled seed 0) -> work/split.json: DEV = first 2000, HOLDOUT = last 1007.
- Practice answer letters: A 354, B 330, C 309, D 356, E 270, F 297, G 280, H 281, I 292, J 238 (true prior is front-loaded).

## Pass 1 (planned): score original on DEV[:512], acts: hidden 0..24, mlp 12..23, o_proj inputs (attn layers), post_operator_norm / post_ffn_norm outputs (all layers). No edit.
