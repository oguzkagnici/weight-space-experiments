Stack: carried gen-8 stack (gen-7 claude L22 two-finalist gate n1786 + gen-7 ed_a L23 short visible answer on freed
duplicate slots) PLUS a new L23 commit-opener unit on n881.

Target behaviour: after a paragraph that narrows the answer to two finalists ("so the answer is either B or D"), the
parent opens the next paragraph with "Thus"/"So" and then continues deliberating (" we", " let", " which option",
" maybe") until the 3,072-token cap cuts the trace (a cut scores 0). The unit makes that opener a decision
("Thus I'll go with (B)" / "Thus answer is (B)"); the parent's L20 wrap-up unit then closes the thinking.

Change (L23 MLP only, on top of the carry):
- Freed n881: it is a near-duplicate of n331/n5105 (h_881 = 1.139 h_331 - 0.158 h_5105, residual 1.4%);
  w2[:,331] += 1.139 w2[:,881], w2[:,5105] += -0.158 w2[:,881].
- n881 gate (w1 row): direction at the L23 MLP input separating "Thus"/"So" after a two-finalist paragraph from 103
  other paragraph-initial openers (LOTO AUC 0.963), bias proxy (mu_hat) with threshold 6.0, prompt-sink subspace
  projected off and penalised; w3 row = mu_hat/|mu|; w2 column (norm 1.43, clipped to Q8_0 block absmax) writes
  " I"/" answer" minus " we"/" let"/" which"/" option"/" what"/" list"/" maybe"/" need".
- Built by code/004_build_opener_unit.py 6.0 0.5 4 2 (fold: code/003).

Evidence (edited-only generate passes on the 96 trace-bank questions, same seeds, paired):
- Teacher-forced (pass 2): at the target openers P(" I"/" answer") 0.096 -> 0.779, P(" we") 0.361 -> 0.029.
  Elsewhere mean |dlogp| 0.00021 per token (99,534 tokens).
- Stack+unit vs parent (bank): all 96 +12.5 pp (SE 4.0, z 3.13), +14/-2, cut 53 -> 43.
  REST48 (no overlap with the fit): +10.4 pp (SE 4.5), +5/-0.
- Unit's own effect, stack+unit vs carry alone (passes 3-6): all 96 +11.5 pp (SE 3.9, z 2.95), +13/-2, cut 47 -> 43;
  REST48 (out of sample) +8.3 pp (SE 4.0, z 2.07), +4/-0; EXTRA16 +12.5 (+2/-0); DEV32 (in-sample: all 22 fit traces)
  +15.6 (+7/-2). 14/96 traces byte-identical to the carry.
- Carry alone vs parent on the same 96: +1.0 pp (SE 3.5), +6/-5 (its gen-8 selection: +5.47, z 2.38).
