# Notes of editor claude, generation 2

- Pass 1 RESULT (x3): paired +1.95 (SE 1.70), +12/-7, 50 predictions changed, dlogp -0.547 (SE 0.083): disruptive, the accuracy gain is noise-level.
- Pass 2: x1.5, both, screen256.
- Pass 2 RESULT (x1.5): paired +1.17 (SE 0.87), +4/-1, 21 changed, dlogp -0.151 (SE 0.031).
- Pass 3: crossover = routers x1.5 (L14, L20 o_proj) + generation-1 candidate claude/equalize_boost (L22, L23 w2; copied unchanged, same parent) on a FRESH DEV set work/screen2.json (256, never used; screen256 was used to choose equalize_boost in generation 1).
- Pass 3 RESULT (routers x1.5 + equalize_boost, screen2): paired +1.56 (SE 1.66), +11/-7, 36 changed, dlogp -0.212 (SE 0.045). Saved work/edit_combo.
- Pass 4: routers x1.5 alone, screen2.
- Pass 4 RESULT (routers x1.5 alone, screen2): +1.17 (SE 1.30), +7/-4, 23 changed, dlogp -0.106. Pooled DEV 512 (screen256 + screen2): +1.17 (SE ~0.8). The combination with equalize_boost (+1.56 on screen2) does not clearly stack. Saved work/edit_r15.
- Pass 5: routers x2, screen2 (dose-response), then choose candidates on DEV.
- Pass 5 RESULT (routers x2, screen2): +1.95 (SE 1.51), +10/-5, 29 changed, dlogp -0.198. Dose-response on DEV: x1.5 +1.17, x2 +1.95, x3 +1.95 (screen256; dlogp -0.55).
- CHOSEN ON DEV (before any HOLDOUT): candidate 1 = routers x2 (work/edit_r2), candidate 2 = routers x1.5 (work/edit_r15). Passes 6-7: x2 on HOLDOUT[0:512]; passes 8-9: x1.5 on the same 512.
