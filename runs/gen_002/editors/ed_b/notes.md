# Notes of editor ed_b, generation 2


## Setup
- Split (code/002): seeded perm (seed 2002) of 3007 practice ids; HOLDOUT = first 768, DEV = rest (2239). In work/split.json.
- Direction (suggested start): is there a correct-vs-incorrect signal at option positions upstream of L22, and does L22.H12 read it via keys?

## Pass 1 (read, parent, DEV[:160]) -- current edit: none
- hidden_states 14,18,20,22,24 all positions; L22 k_norm outputs; attn_last all attention layers.
- Results pass 1 (code/004-009): parent acc 43.1% on these 160. L22.H12 attends mostly to option END tokens (0.65 of its mass; letter tokens 0.12).
  H12 attention argmax picks gold 40.6%; agrees with the model's final pick 72.5%. No positional prior of note in H12 attention (argmax counts ~ gold counts);
  the A/B/C-up, I/J-down letter bias arises AFTER selection (final centered logits A .54 B 1.13 C 1.07 ... I -1.51 J -1.63).
- Correctness probes at option ends (5-fold CV by question, within-question centered): diff-of-means h14 40.0%, h18 32.5%, h20 30.6%, h22 25.6%; LR(C=1e-3) h14 37.5, h20 41.2, h22 35.0.
  On MODEL-WRONG questions the probes pick gold 8-12% (chance ~10%); span-mean probes 13-16%. => no linear unread correctness signal at option positions.
  Ensembles final logits + b*probe or + b*log attn(H12) give <= +2/160 questions. NEGATIVE for "strengthen the key read": H12 already reads what the options carry.
- Next: localize WHERE (layers 4..16) the option-end belief is written and whether a specific head's contribution is mostly signal (amplify -> SNR).

## Pass 2 (read, parent, 180 DEV[160:700] questions <=320 tokens, list in work/p2_qids.json) -- current edit: none
- hidden_states 4,8,10,12,14,16 all positions; o_proj inputs L8,10,12,14; attn_last L22.
- Results pass 2 (code/012-015): option-end belief forms between h8 and h14 (sep along pass-1 d14 direction: h4 .02, h8 .25, h10 .81, h12 .89, h14 1.00, h16 .99; CV-DM acc h12 33%, h14 31%; H12 acc on this set 32%).
  Heads writing it: L12.H13 (diff .50, SNR .85, alone 29%), L10.H4, L8.H1, L12.H8 -- but they correlate 0.5-0.83 with the rest of the stream; linear amplification screen (x0..x5) changes belief acc by <= +1/180. The bulk (diff ~4) is written by MLP/conv; signal is one coherent, redundant belief.
  L14.H12 is already a selection head at option ends (argmax acc 41.9% on pass-1 set, mass .53); every selecting head is at chance on model-wrong questions.
  CONCLUSION: the content path is saturated; errors are knowledge (gold at chance among wrong answers). Abandon the key-path direction.
- Pivot: the post-selection letter stage (H12 attention argmax counts are flat vs gold, final picks are A/B/C-heavy).

## Pass 3 (score, parent, DEV[700:1212], 512 q) -- current edit: none
- activations: hidden 18,20-24; mlp 20-23; o_proj input L22.
- Results pass 3 (code/017-024), 512 DEV[700:1212], parent acc 34.2%: pred A77 B96 C77 D50 E47 F56 G38 H33 I20 J18 vs gold A49 B67 C41 D57 E50 F60 G39 H53 I49 J47.
  Final-logit decomposition (10-opt, centered): prior at h22 A+1.0 B+1.2 .. J-.7; L22.H12 copies with its own late-letter tilt (A -.75 B -1.44 G +1.27); mlp22 re-adds early tilt (A +.82 .. J -.83).
  Stage accuracies (lens): h22 24.2, +attn22 34.4, +mlp22 32.4, +conv23 34.2, final 34.8; H12 alone 34.0.
  2-fold CV constant debias ceiling: only +0.6 pt on this set (34.8 -> 35.4). Lens removal/scaling of mlp22, mlp23, conv23, non-H12 L22 heads: all within +-1 pt.
  Surface heuristics: model over-picks long options (pred longest 26% vs gold 19%), but a CV length correction gives no gain. Lexical overlap: no effect after tie-aware ranks.
  L14.H12 & L22.H12 agree 63%; ensembles <= +2/160.
  => Errors are knowledge-limited; no large repairable mechanism visible. Remaining passes: causal tests of the content path (nonlinear effects lens cannot predict) + pick best.

## Pass 4 (both, S1 = DEV[1212:1468], 256 q) -- edit X1: model.layers.12.self_attn.o_proj.weight columns 1664:1792 (head 13) x2.0
- Result pass 4 (X1, L12.H13 x2) on S1: parent 28.9%, edited 30.5%: acc +1.56 (SE 1.46), 9 w->r / 5 r->w, dlogp(correct) -0.319 (SE .044).
  Paired analysis (code/026): mean gold rank unchanged (diff +0.004 +- 0.055); entropy .99 -> .90; dlogp on orig-wrong -0.56, orig-correct -0.01.
  => doubling the strongest belief writer only SHARPENS the output; it does not improve the ranking (as the lens predicted). Saved in work/best_x1.
- code/027: L22 MLP leader-suppressor bank (pass 3): when X leads after attn22, its suppressor DLA on X: A n389 -.76, B n6620 -.61 + n5859 -.43, C n1812 -1.73, D n464 -3.43, E n7357 -4.13, F n6492 -3.09, G n3344 -3.56, H n397 -3.35, I n2786 -3.97, J n3310 -3.15 (matches g1). L23 bank weaker (-1.0..-1.7) and more even.
- code/028 lens screen of equalizing this bank (down C-J x.5 / up A-C / all to -2 logits): acc .326-.346 vs base .348; no variant gains. Output stage is near-optimal for accuracy on this set.

## Pass 5 (both, S1) -- edit X0: model.layers.12.self_attn.o_proj.weight columns 1664:1792 (head 13) x0.0 (ablation, dose-response of the content path)
- Result pass 5 (X0, L12.H13 ablated) on S1: acc +1.17 (SE 1.70), 11 w->r / 8 r->w, 69 preds changed, dlogp(correct) +0.564 (SE .082).
  Gold rank diff -0.020 +- 0.076; entropy .99 -> 1.24; dlogp orig-wrong +0.90, orig-correct -0.16. Preds shift A 39->24, G 26->42.
  => Dose-response x0 / x1 / x2: entropy 1.24 / .99 / .90, gold rank unchanged throughout. L12.H13 sets the CONFIDENCE of the option belief, not its ranking.
  The content path is causal for confidence but redundant for which option wins.
- Decision (DEV only): no rank-improving edit found. Candidates = X1 (DEV +1.56) and X0 (DEV +1.17). Measure on HOLDOUT: X1 on HOLD[0:768] (3 passes), X0 on HOLD[0:512] (2 passes).

## Pass 6 (both, HOLD[0:256]) -- edit X1: L12 o_proj cols 1664:1792 x2.0
- Result pass 6 (X1 on HOLD[0:256]): parent 37.1%, acc -0.78 (SE 1.24), 4 w->r / 6 r->w, dlogp -0.305 (SE .041). As predicted: a confidence knob, no accuracy gain.
- Revised plan: X0 and X1 each on HOLD[0:512] (2 passes each); 1 read pass to test whether the model has knowledge the MC format does not use (option-likelihood "cloze" scoring on model-wrong questions).

## Pass 7 (both, HOLD[0:256]) -- edit X0: L12 o_proj cols 1664:1792 x0.0
- Result pass 7 (X0 on HOLD[0:256]): acc -1.56 (SE 1.91), 10 w->r / 14 r->w, dlogp +0.470 (SE .075). Again a confidence knob only.

## Pass 8 (read, parent, cloze) -- 108 questions from the pass-3 set (prompt <=250 tokens), 1024 texts = question + "Answer with the correct answer." + assistant "The answer is: <option text>"; token_logprobs.
- Result pass 8 (cloze, code/030; pass was interrupted by a machine restart and requeued, counted once): 108 questions, MC acc 33.3%.
  Option-text likelihood (mean per token) picks gold 25.9% overall, 25.0% on MC-WRONG questions (18/72; chance ~10%), 27.8% on MC-right; sum-logprob 20.4%.
  => the model holds some answer knowledge that the letter-selection circuit does not use (cloze correctness ~ independent of MC correctness).
  Ensemble MC logprob + 0.25*z(cloze) 35.2% vs 33.3% (+2/108; small n). Lead for next generation, not exploitable in my remaining budget.

## Pass 9 (both, HOLD[256:512]) -- edit X1: L12 o_proj cols 1664:1792 x2.0
