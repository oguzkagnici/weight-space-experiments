# Generation 14 report

- Created: 2026-11-16 17:09 UTC
- Parent: `gen008_carry_stack_shortans_L23_finalist_L22`
- Editors: claude, ed_f; budget 8 passes each; up to 1 candidates each
- Hidden selection subset: 256 questions (seed `selection-gen-14`)

## Suggested starting points

- **claude**: STEM-CHECK CIRCUIT (inside the network). Follow the order of THIS GENERATION'S TARGET. Your emphasis is step 1(b) and step 3: on >= 40 mixed verbal pass@4 questions, find the option-judging positions and measure attention from them to the stem tokens per layer and head (`attn_last` on prefixes), right vs wrong samples of the same question. Look for heads whose stem attention differs within question, and check whether they carry the decisive cue (e.g. ablate or patch their output at judging positions and see whether P(right conclusion) moves). Run the causal insertion test (step 2) before any edit; if it is null, report the negative result rather than editing.
- **ed_f**: STEM CUE AT THE TEXT AND DECISION LEVEL. Follow the order of THIS GENERATION'S TARGET. Your emphasis is step 1(a) and step 2, done at scale as you did in generation 13: build an automatic measure of stem re-statement while options are judged, compare right vs wrong samples of the same question on as many mixed verbal questions as the bank allows, and run the causal insertion test (cue quote vs equal-length control) on many wrong samples. Only if quoting the cue moves the conclusion, find what makes the trace consult the stem (e.g. the token positions and components that start a stem re-statement in right samples) and strengthen it where the right samples have it. Judge only on the holdout bank; your generation-13 dev set picked from parent failures was misleading.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| claude | 1 | read | original | 120 | read 120 texts | 1226.4 |
| claude | 2 | read | original | 98 | read 98 texts | 1063.5 |
| claude | 3 | read | edited | 120 | read 120 texts | 1304.0 |
| claude | 4 | generate | edited | None | read None texts | 2026.3 |
| claude | 5 | generate | edited | None | read None texts | 1763.8 |
| ed_f | 1 | read | original | 110 | read 110 texts | 1907.4 |
| ed_f | 2 | read | original | 90 | read 90 texts | 1558.2 |
| ed_f | 3 | read | original | 474 | read 474 texts | 1442.5 |
| ed_f | 4 | read | original | 139 | read 139 texts | 1107.2 |
| ed_f | 5 | read | edited | 159 | read 159 texts | 1595.5 |
| ed_f | 6 | generate | edited | None | read None texts | 1902.2 |
| ed_f | 7 | generate | edited | None | read None texts | 1852.5 |
| ed_f | 8 | read | original | 60 | read 60 texts | 1069.5 |

## Candidates

### claude/claude_g14_stemcheck_L22H10_amp

- Tensors: model.layers.22.self_attn.o_proj.weight
- Weights changed (bf16): 262144

# claude_g14_stemcheck_L22H10_amp: NEGATIVE RESULT for the stem-check line; minimal measured edit
Edit: model.layers.22.self_attn.o_proj.weight, columns 1280:1408 (the 128 inputs of head 10) x1.5. Nothing else.
Target: the STEM CHECK (do wrong samples fail to consult the decisive cue in the question stem while judging options?).
Evidence that the line is closed (all on practice questions, parent):
- Text level (31 finished-wrong verbal questions; 15 have a decisive stem cue): every wrong sample USES the decisive cue, usually many times
  (e.g. 6110 'informed' 18-50x, 1409 'WEAKEST' 20x, 6614 'weight/BMI' 37-62x); within 5 mixed cue questions right samples mention it only ~20-30% more.
  Wrong samples see the cue and draw the wrong inference from it; the other ~half of finished-wrong verbal questions are pure recall with no cue.
- Network (pass 2; 98 judging-position prefixes of 35 samples, 10 mixed verbal questions, attn_last at 11 layers): attention from judging positions
  to the stem is ~1% of the mass and slightly HIGHER in wrong samples (.0113 vs .0096 at the gold option's first evaluation); within-question
  right-wrong differences: no head |z| > 3 of 176; split-half correlation of head differences -0.07 / +0.06 (no replicable pattern).
- CAUSAL INSERTION GATE (pass 1; 32 finished-wrong samples of the 15 cue questions): inserting 'Let me re-read the key part of the question: "<cue>"'
  vs the same frame quoting a non-decisive stem sentence, then 'The answer is (': P(gold) among letters +0.041 (SE 0.039) early (~30% of the trace),
  +0.085 (SE 0.066) late (commit paragraph, 12 samples); vs no insertion -0.001 (SE 0.066). Top-1 becomes gold in +2/32 and +1/12.
  Quoting the cue does not change the conclusion, so no edit that makes the model attend to it can help.
Edit rationale: L22.H10 is the only head whose right>wrong stem attention replicated in both halves of the questions (weak, z 2.75 / 1.31).
Read effect of the edit (pass 3, the 120 pass-1 texts, paired): |dP(gold)| <= .0033 in every condition (SE .0015-.0035), top-1 unchanged: null.
HOLDOUT48 at 8,192 (passes 4-5): overall 62.5 -> 60.4% (-2.08 pts, SE 6.31; wrong->right 4, right->wrong 5);
both-finish (n 38) 73.7 -> 71.1% (-2.63, SE 5.95; wrong->right 2, right->wrong 3); cuts 7 -> 8; 0/48 traces identical, first difference at a median
~2% of the trace: the edit only re-draws the samples (resampling noise). Expect ~0 on selection.

### ed_f/ed_f_g14_stem_attention_at_judging_L22

- Tensors: model.layers.22.self_attn.q_proj.weight
- Weights changed (bf16): 733127

# ed_f_g14_stem_attention_at_judging_L22: "look at the question stem while an option is judged" (layer-22 queries, position-gated). Submitted as the measured weight-level test of the STEM CHECK; the diagnosis is NEGATIVE and the expected selection result is about 0.

TARGETED BEHAVIOUR. Generation 13 suggested that wrong samples do not go back to the decisive cue in the question stem when they judge the options. This edit raises the attention of six layer-22 heads to the stem tokens at option-judging positions.

WHAT WAS CHANGED. model.layers.22.self_attn.q_proj.weight, rows of heads 1, 11, 3, 0, 8, 15 (128 rows each; the six L22 heads with most stem attention at judging positions, .044-.057): W_q[h] += gam_h * (t_g / w_q) (x) d, one rank-one term per head, row change 1.2-1.7%.
- d (query side, 2,048-d): unit mean-difference direction of the layer-22 attention input between option-judging positions (the label / "label option-text:" that opens a judging paragraph) and random trace positions, orthogonal to the background mean; fitted on half of the questions. d.x = 1.41 at fit judging positions, 1.20 at held-out ones, 0.21 at held-out background (AUC .92), 0.02 on prompt tokens.
- t_g (key side, 128-d): unit "stem key" direction of kv head g = h // 4: mean normalised key of stem tokens minus the mean of option-token and trace-token keys (t.k = +5.0 stem, +0.9 options, -4.4 trace on held-out questions). Layer 22 has no positional rotation: q_norm(W_q x).k_norm(W_k x)/sqrt(128) reproduces the recorded attention (corr .93-1.00 for 15 of 16 heads), so the edit is computed exactly from recorded activations (means and differences only; no fitting to a loss).
- gam_h: set for +2.2 log-odds toward a stem key at judging positions.

EVIDENCE FOR THE DIAGNOSIS (parent, read passes; 288 parent traces at 8,192).
(1) Text level: wrong samples re-state the stem as much as right samples of the same question (10 mixed verbal questions: stem-only words 61 vs 64 per 1,000, within-question -3.3, SE 3.6; stem quotes +0.1, SE 0.5; in the 256 tokens before the first commitment wrong samples re-state MORE: stem 3-grams 11.5 vs 40.5, SE 10.6). The decisive cue's key words are as frequent in wrong as in right samples.
(2) Attention from option-judging positions to the stem (64 texts, 9 questions): right .0274 vs wrong .0240; with question, length and format controls +.0013 (SE .0018). No head differs beyond chance (1 of 49 with |z| > 2). The stem gets 2.4% of full-attention-layer attention there; no head is a dedicated stem reader, and none reads the cue selectively (cue share of stem attention 6-17%, its share of the stem tokens).
(3) CAUSAL INSERTION TEST (the read gate of step 2). 30 finished-wrong samples of the 13 questions that have a decisive stem cue (of 34 questions with a wrong sample), cut before the paragraph that first commits to the wrong letter; insertion 'Wait, let me re-read the question. It says: "<phrase>".' then a forced "So the answer is (". CUE quote vs an equal-length NON-cue quote of the same stem, clustered by question: margin log P(gold) - log P(wrong pick) -0.14 (SE 0.81); P(gold) +.096 (SE .070); forced conclusion = gold 2/30 -> 5/30 (+.147, SE .090). Replication with another wording and other control phrases: margin +1.26 (SE 1.39); gold 0/30 -> 6/30 (+.218, SE .106); per-question effects correlate r .79. All gains are on 4 health/psychology vignettes (2654, 6110, 6614, 6800: +2.3 to +11.5 logits); on the law/history questions the cue quote makes the wrong pick firmer (1275 -5.4/-3.7, 1705 -3.6/-2.6, 1643, 1196). Adding one sentence with the cue's implication (oracle, names no option) gives 10/30 (late cut) and 14/30 (early cut; +.34 over the cue alone, SE .13): what is missing is the inference, not the look at the stem. Any re-read insertion shakes the commitment (margin +7.8 over no insertion, SE 1.1) without changing the pick (1/30 -> 2/30). A cue quote leaves right conclusions alone (6/7 -> 6/7). After a forced re-read opener the model would not quote the cue itself (log P of the first cue token -15.9 vs -7.6 for another stem phrase).

READ GATE OF THE EDIT (edited vs parent on the same texts; questions NOT used to fit d or t). Stem attention of the six edited heads at 57 held-out option-judging positions .048 -> .108 (+.061, SE .008); other L22 heads .029 -> .029; held-out background positions .123 -> .139; conclusion probes .047 -> .080. The next-token distribution at judging positions barely moves (total variation .019; top-1 changed in 1 of 89); forced conclusions on the 30 late-cut wrong samples: margin +0.07 (SE 0.03), gold 1 -> 1. Off-target over 181k teacher-forced tokens: mean |dlogp| .0088, 1.7% of tokens > .1.

HOLDOUT48 at 8,192 tokens (paired with the parent bank, 48 questions, two edited-only passes): overall 62.5 -> 60.4, -2.08 points (SE 5.56; wrong->right 3, right->wrong 4). BOTH-FINISH (39 questions): 71.8 -> 71.8, +0.00 (SE 5.20; wrong->right 2, right->wrong 2; 8 of 39 answers changed). Cuts 7 -> 7. 1 of 48 traces identical; first difference at a median 4.1% of the trace, before the first judging paragraph in 28 of 29 traces. In the generated traces explicit references to the question in the judging region rise from .78 to 1.20 per 1,000 words (SE .19); conclusions do not change. GSM8K was not measured.

CONCLUSION. The stem check is a symptom, not a mechanism: wrong samples consult the stem as much as right ones, the network has no component that brings the decisive cue into option judging, and more attention to the stem (2.2x where options are judged) changes no conclusions.

## Selection (hidden questions) and decision

Parent accuracy on the 256 selection questions: **60.55%**. Rule: z >= 2.0; both-finish z >= 1.0; gsm8k_boxed guard z > -2.0.

| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |
|---|---|---|---|---|---|---|---|---|
| ed_f/ed_f_g14_stem_attention_at_judging_L22 | 62.11 | +1.56 | 2.1 | +0.74 | 16 | 12 | +0.0000 | 733127 |
| claude/claude_g14_stemcheck_L22H10_amp | 59.77 | -0.78 | 1.9 | -0.41 | 11 | 13 | +0.0000 | 262144 |

**Decision:** no candidate passed; the parent survives.

## Editors' reports

### claude

_(no report yet)_

### ed_f

_(no report yet)_

