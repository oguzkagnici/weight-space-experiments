# Notes of editor ed_a, generation 10


## Diagnosis (code/001-006, passk 8192 bank + 3072 bank)
- First preference statement is late (rel pos >0.3 in 208/214 finished traces) and = final answer (133/142 right, 64/72 wrong). No early anchoring.
- Wrong finished answers state two finalists more often (42/83 = 51%) than right ones (36/180 = 20%).
- When gold and pred are both in the last stated pair: gold is the EARLIER letter -> right 15/15 (10 questions); gold the LATER letter -> right 14/27 (wrong in 9 questions). Primacy bias in finalist choice (letter label or discussion order).
- Overall wrong preds earlier than gold in 22.1 of 29 questions (19.0 expected): weak.

## Pass 1 (read, original): forced-commit probes
- 82 texts = prompt + think up to end of line with last pair statement + "\n\nThus I'll go with (" (code/007-008, work/probe_sel.json). topk 50 at last, hidden 14-24, mlp 14-23, o_proj inputs 14/18/20/22, attn_last 18/20/22.
- Edit: none (original).
