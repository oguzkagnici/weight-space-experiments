# Notes of editor claude, generation 3

## Generation 3 (parent = base). Starting point: the belief-forming writers in layers 8-14 at the option-end tokens.
- Same DEV/HOLDOUT split as my earlier generations (work/split.json). 192 DEV questions (seed 31), option-end token positions (last content token of each option line) in work/option_ends.json (code/001).
- Pass 1: read original, residual stream after layers 7-15 at all positions (38,930 tokens).
- Passes 2-5 RESULT (DEV 512 = screen256 + screen2): x0 (full ablation) -1.37 (SE 1.45), +24/-31, gold rank +0.076 (SE 0.057, worse), dlogp +0.40; x0.5 +0.39 (SE 0.87), +11/-9, gold rank -0.002 (SE 0.027), dlogp +0.19. The generation-2 hidden gains (+1.25, rank -0.11) do NOT replicate on my DEV: likely winner's curse; the calibration gain replicates.
- CHOSEN ON DEV: primary x0.5; x0 submitted as an explicit re-test of the generation-2 result. Passes 6-7: x0.5 on HOLDOUT[0:512]; passes 8-9: x0 on the same.
