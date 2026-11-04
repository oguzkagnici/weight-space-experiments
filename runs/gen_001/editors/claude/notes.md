# Notes of editor claude, generation 1

## Setup
- DEV/HOLDOUT split of practice: work/split.json (seed 1234; HOLDOUT = 1,000 questions, DEV = 2,007). HOLDOUT is used only for confirmation.
- Pass 1 (planned): diagnostic score of the original on 512 DEV questions (37 per subject), with the residual stream after layers 8-24 and the per-head attention outputs (o_proj inputs, 16 heads x 128) of all 11 attention layers at the answer position.
