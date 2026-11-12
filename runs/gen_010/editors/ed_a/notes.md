# Notes of editor ed_a, generation 10


## Diagnosis (code/001-006, passk 8192 bank + 3072 bank)
- First preference statement is late (rel pos >0.3 in 208/214 finished traces) and = final answer (133/142 right, 64/72 wrong). No early anchoring.
- Wrong finished answers state two finalists more often (42/83 = 51%) than right ones (36/180 = 20%).
- When gold and pred are both in the last stated pair: gold is the EARLIER letter -> right 15/15 (10 questions); gold the LATER letter -> right 14/27 (wrong in 9 questions). Primacy bias in finalist choice (letter label or discussion order).
- Overall wrong preds earlier than gold in 22.1 of 29 questions (19.0 expected): weak.

## Pass 1 (read, original): forced-commit probes
- 82 texts = prompt + think up to end of line with last pair statement + "\n\nThus I'll go with (" (code/007-008, work/probe_sel.json). topk 50 at last, hidden 14-24, mlp 14-23, o_proj inputs 14/18/20/22, attn_last 18/20/22.
- Edit: none (original).

## Pass 1 result (code/009, 011, 012)
- Forced commit right after the line with the last finalist pair: the letter choice is mostly already decided (P>0.9 for one letter in most probes); near-ties (0.2-0.8) in only ~6/82.
- gold earlier: probe picks gold 16/17 (mean P(gold) .80); gold later: 13/27 (mean P(gold) .52). Mean logit(earlier)-logit(later) +0.87; P(earlier wins) .65.
- Final letter logits of NON-finalist letters fall by 0.195 per letter step (A->J ~1.75 logits): a content-free early-letter ramp at thinking decision points.
- DLA: consistent ramp writers (low sd across probes): L20 n1000, L22 n2820, L17 n616, L17 n2401, L23 n4597, L15 n4408, L20 n3964, L19 n6621, L22 n2389, L19 n6612, L20 n3142 (sum -0.247/letter in direct read-out). Several are the phase-1 prior writers.

## Edit v1 (code/013, F=0.65): letter-ramp removal
- For each of the 11 neurons above: w2[:,j] += -0.65*slope_j*u_j, u_j = headroom-weighted minimal-norm residual direction whose read-out (post_ffn gain x final-norm gain x centred letter embeddings) is the pure ramp (idx-4.5); entries kept within 0.98x Q8 block absmax. Each column's ramp slope reduced to 35%.
- Pass 2: read edited on the 82 probes (topk, h24). Pass 3: generate edited cap 8192 n=4 on 6110, 5325, 2055, 6941, 2654, 6562 (gold-later finalist losses).
