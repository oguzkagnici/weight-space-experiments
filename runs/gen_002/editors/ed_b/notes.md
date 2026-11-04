# Notes of editor ed_b, generation 2


## Setup
- Split (code/002): seeded perm (seed 2002) of 3007 practice ids; HOLDOUT = first 768, DEV = rest (2239). In work/split.json.
- Direction (suggested start): is there a correct-vs-incorrect signal at option positions upstream of L22, and does L22.H12 read it via keys?

## Pass 1 (read, parent, DEV[:160]) -- current edit: none
- hidden_states 14,18,20,22,24 all positions; L22 k_norm outputs; attn_last all attention layers.
