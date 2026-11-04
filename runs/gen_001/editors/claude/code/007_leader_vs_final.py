# hypothesis H2 consequence: for late-letter leaders (entering the L22 MLP), the leader is right more often than the
# final answer (suppression wrongly removes them); for A/B leaders the reverse or equal. Pure bookkeeping on recorded data.
import json, numpy as np, torch
from wsp.editor import *
exec(open("code/006_letter_suppressors.py").read().split("rep = []")[0].split("print(f\"lens top")[0])
print("leader letter | n | leader correct | final correct | final kept leader")
for L in range(10):
    m = top_pre22 == L
    if m.sum() < 5: continue
    print(f"  {'ABCDEFGHIJ'[L]} | {m.sum():3d} | {np.mean(ans[m] == L):.2f} | {np.mean(pred[m] == ans[m]):.2f} | {np.mean(pred[m] == L):.2f}")
early = top_pre22 <= 1; late = top_pre22 >= 4
for name, m in (("A/B leaders", early), ("E-J leaders", late)):
    print(f"{name}: n {m.sum()}, leader acc {np.mean(ans[m] == top_pre22[m]):.3f}, final acc {np.mean(pred[m] == ans[m]):.3f}")
