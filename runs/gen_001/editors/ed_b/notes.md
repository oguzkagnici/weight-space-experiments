# Notes of editor ed_b, generation 1


## Setup
- code/001_split.py: practice split stratified by category, seed 1234 -> work/split.json: DEV 1499, HOLDOUT 1508.

## Pass 1 (diagnostic)
- Edit: none. model=original, questions = DEV[:512]. Activations: hidden states 0..24, mlp 14..23, post_operator_norm / post_ffn_norm outputs 12..23, o_proj inputs for attn layers 12..22.
