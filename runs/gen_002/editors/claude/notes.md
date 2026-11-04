# Notes of editor claude, generation 2

- Pass 1 RESULT (x3): paired +1.95 (SE 1.70), +12/-7, 50 predictions changed, dlogp -0.547 (SE 0.083): disruptive, the accuracy gain is noise-level.
- Pass 2: x1.5, both, screen256.
- Pass 2 RESULT (x1.5): paired +1.17 (SE 0.87), +4/-1, 21 changed, dlogp -0.151 (SE 0.031).
- Pass 3: crossover = routers x1.5 (L14, L20 o_proj) + generation-1 candidate claude/equalize_boost (L22, L23 w2; copied unchanged, same parent) on a FRESH DEV set work/screen2.json (256, never used; screen256 was used to choose equalize_boost in generation 1).
