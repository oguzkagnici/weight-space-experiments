# Hypothesis: none (submission). Submit E1+E2 (primary) and E1 alpha=0.5 (second) from the saved edit folders.
import shutil, os, glob
from wsp.editor import *
def stage(src):
    for f in glob.glob("edited/*"): os.remove(f)
    for f in glob.glob(src + "/*"): shutil.copy2(f, "edited/")
    print("staged", edited())
D1 = """Removes the option-ordinal ("primacy") query component from the answer-selecting attention layers L22 and L20.
Change: model.layers.22.self_attn.q_proj.weight and model.layers.20.self_attn.q_proj.weight, all 16 heads. For head h: W_h <- W_h - t t^T W_h (alpha = 1), where W_h = rows h*128:(h+1)*128 and t = normalize(q_norm.weight * v1_g). v1_g is PC1 of the slot-mean key vectors (k_norm outputs averaged over the tokens of options A..J, 120 DEV questions) of kv group g = h//4. Rank-1 per head, 2 tensors.
Mechanism: L22 is NoPE-like (q.k on recorded q_norm/k_norm reproduces the attention logits, residual var 0.3%). Option keys carry an ordinal position signal along v1 (slot means A -5.7, B -3.8, ..., J +0.3; split-half r .97). The answer-position query has a near-constant large component along v1 (about 32% of |q|), so every option-reading L22 head prefers early options by about 2.2 nats (A vs I). L20 (RoPE, but v1 sits in low-frequency pairs) has the same mechanism with an A-first preference of about 1 nat. Removing it flattens the letter prior: centred I goes from -2.1 to -0.6.
Evidence (paired edited - parent): DEV S1 (DEV[512:768], 256 q): +1.95 (SE 1.87), logp(correct) +0.186 (SE 0.060). HOLDOUT 512 practice questions (never used for any choice): +0.39 (SE 1.46), 29 w->r / 27 r->w, logp +0.197 (SE 0.055). The log-prob gain is robust, but the accuracy gain is not significant. A becomes under-picked (gold-A accuracy .43 -> .22)."""
D2 = """Half-strength version, L22 only: model.layers.22.self_attn.q_proj.weight, all 16 heads, W_h <- W_h - 0.5 t t^T W_h, where t = normalize(q_norm.weight * v1_g) and v1_g = PC1 of the L22 slot-mean option keys of kv group h//4. This halves the constant query component that makes L22 heads prefer early options (primacy bias A > ... > J of about 2.2 nats). Rank-1 per head.
Evidence (paired): DEV S1 (DEV[512:768], 256 q): +1.95 (SE 1.03), 6 w->r / 1 r->w, logp +0.129 (SE 0.031). HOLDOUT 512: -0.59 (SE 0.98), 11 w->r / 14 r->w, logp(correct) +0.115 (SE 0.026). The logp gain is robust; there is no accuracy gain."""
stage("work/edit_E1E2"); print(submit_candidate("ed_a_qprimacy_L22L20", D1))
stage("work/edit_E1_a0.5_final"); print(submit_candidate("ed_a_qprimacy_L22_half", D2))
