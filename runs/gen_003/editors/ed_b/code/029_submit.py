# Hypothesis: none (submission). Submit candidate 2 (current edit: all L22 heads letter-token query projection), then rebuild and submit candidate 1 (H12 only).
import subprocess, sys
from wsp.editor import submit_candidate, edited
which = sys.argv[1]
if which == "all":
    d = ("Removes the content-free 'option letter token' vote from all layer-22 attention queries. Change: model.layers.22.self_attn.q_proj.weight only; for every head h (16 heads, rows h*128:(h+1)*128): W_h <- W_h - a w_g w_g^T W_h, rank 1 per head, a = 1.0 (head 12: a = 1.5), "
         "w_g = normalize(q_norm.weight * u_g), u_g = mean over 240 DEV questions of (mean k_norm key at the option letter tokens '(X' minus mean key at the option end and newline tokens) for kv group g = h//4. "
         "Mechanism: L22 is NoPE (q.k reproduces attention); its answer-copy head H12 puts 12% of its attention on the letter tokens '(X', whose keys precede the option text (causal) and so carry only a slot code, not the option's content; many other L22 heads do the same (7-50%). Lens (DEV, recorded attention): dropping letter tokens raises H12's option pick +5/-2 and that of H0/H3/H11 by +8 to +20 net; q_norm renormalizes the query after the projection. "
         "Evidence (paired, edited - parent): DEV3 256 q: acc +0.39 (SE 1.17), 5/4, dlogp(correct) +0.040 (SE 0.023), gold rank -0.078 (SE 0.039). HOLDOUT 512 q (measured once, after choosing on DEV): acc 0.00 (SE ~0.74), 7 wrong->right / 7 right->wrong, dlogp(correct) +0.047 (SE ~0.017), gold rank -0.031 (SE ~0.027). Accuracy-neutral; small replicated calibration gain.")
    print(edited()); submit_candidate("letterproj_L22_all", d)
else:
    subprocess.run([sys.executable, "code/027_make_edit_letterproj.py", "1.5", "12"], check=True)
    d = ("Removes the content-free 'option letter token' vote from the answer-copy head L22.H12 only. Change: model.layers.22.self_attn.q_proj.weight rows 1536:1664 (head 12): W_h <- W_h - 1.5 w w^T W_h (rank 1), "
         "w = normalize(q_norm.weight * u), u = mean over 240 DEV questions of (mean group-3 k_norm key at the option letter tokens '(X' minus mean key at the option end and newline tokens). "
         "Mechanism: L22 is NoPE; H12 (the head whose attention picks the answer, +2.9 logits to gold) puts 12% of its attention on letter tokens whose keys precede the option text (causal), i.e. a content-free slot vote (B over-picked). Lens: dropping letter tokens from H12's option vote +5/-2 (recorded attention) / +8/-0 (recomputed q.k), letter-token mass .116 -> .026 at this strength. "
         "Evidence (paired): DEV3 256 q: acc +1.17 (SE 0.87), 4/1, dlogp +0.022 (SE 0.023), gold rank -0.055 (SE 0.037). HOLDOUT 512 q (measured once): acc 0.00 (SE ~0.73), 7/7, dlogp +0.020 (SE ~0.017), gold rank -0.025 (SE ~0.027). Accuracy-neutral.")
    print(edited()); submit_candidate("letterproj_L22_h12", d)
