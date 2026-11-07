# Notes of editor ed_a, generation 7

## Plan (start)
- Goal: an edit that is BITWISE silent off-target in llama.cpp Q8_0 (det1, REPACK/LLAMAFILE off ->
  ggml_vec_dot_q8_0_q8_0: activations entering w2 are quantised to Q8_0 per block of 32 neurons).
  => a neuron j contributes exactly 0 on a token iff round(h_j / (amax_block/127)) == 0, i.e. |h_j| < amax_block/254.
  A repurposed neuron is exactly silent on a token if the parent's h_j AND the edited h'_j both quantise to 0
  there (block amax then unchanged, other 31 values identical). The parent's own L20 n5414 is already such a gated
  neuron, so changes to n5414 act only where it is (or becomes) nonzero after quantisation.
- Targets considered: (1) commit moments n5414 misses (suggested start); (2) silent version of gen6 short-answer.

## Findings before pass 1 (code/001-003)
- Bank: 53 cut; 5 of them closed the think and were cut in the visible answer (1740 D ok, 9274 B ok, 8198 C ok,
  6829 F wrong, 6014 no stated answer). Mid-trace "answer is" statements in cut traces are mostly hedged two-option
  lines ("Thus likely answer is G or D"); firm+correct mid statements: 8932 only (+6941 "likely intended J").
- code/003: no MLP neuron in any layer has a Q8_0 w2 column that quantises to zero (min 1934 of 2048 nonzero).
  So exact silence must come from ACTIVATIONS quantising to 0 (|h_j| < block amax/254).

## Pass 1 (read, original = parent), work/req1.json
- 15 bank traces (code/004: 6 targets 1740 9274 8198 6829 6014 8932 + 9 finished, one per category), 35,806 tokens.
- return token_logprobs, mlp [20,21,22,23] all positions, inputs of w1 at L20, L22, L23.
- Purpose: (1) per-neuron fraction of tokens with nonzero Q8 activation (quiet-neuron screen), (2) gate design at </think>.
- code/005: finished bank traces: visible-answer letter == think's last stated letter in 41/43 (7811 no stated
  letter but correct C; 5603 no answer). Tokens: '"The'=14225 ' answer'=5141 ')".'=94078 <|im_end|>=124900 </think>=124902.

## Plan A: exactly-silent short visible answer (L23 hosts)
- An L23 neuron changes only the logits at its own position (nothing reads L23's output later), so teacher-forced
  parent reads of '...</think>"The answer is (X)".<|im_end|>' give exactly the inputs the edited model sees.
- Need 3 L23 neurons whose parent activations quantise to 0 on all thinking tokens, and gates that are <<0 (silu~0)
  off-target: S1 at </think> -> '"The'; S2 at '"The' after </think> -> ' answer'; S3 at ')".' after it -> <|im_end|>.

## Pass 2 (read, original), work/req2.json (code/006), queued while pass 1 waits
- 46 synthetic texts (prompt+think+'</think>"The answer is (X)".<|im_end|>', X = think's last letter; 41 finished +
  closed-cut) + 6 cut traces; 92,954 tokens. return token_logprobs, topk 10, mlp [23], input of L23 w1, all positions.

## Pass 1 results (work/fwd_01) -- code/007
- NO neuron in L20-L23 is Q8-silent: per neuron, the activation quantises to nonzero on 33-95% of tokens
  (median 92-95%; even the parent's gated n5414 is nonzero on 60% of tokens; quietest = L22 block 4512-4543, ~35%).
  So bitwise silence of a repurposed neuron is impossible here. Revised goal: MINIMAL perturbation.
  Key: an L23 neuron's output only reaches the logits at its own position (no later layer, no quantisation
  amplification), so the off-target effect = removal of the host's original logit contribution + gate leak.
  Gate leak can be exactly 0 (g << -20 -> h' ~ 1e-8 quantises to 0). Choose hosts with the smallest logit impact.
- code/009: L23 per-neuron logit impact (rms h x vocab-spread of w2 col through norms): quietest 2e-4 (4526, 135,
  74, 810 = the gen5/6 hosts), median 6e-4 -> only 3x quieter than average; explains gen6's early divergence.
- code/010-011: L23 contains DUPLICATE neurons: cluster {1228, 3401, 6986, 3793} (cos w1 .9998, w3 +-.9999, w2 +-.9998,
  activation corr .9999), {331, 5105, 881} (.9995), {5821, 2400} (.997). Freeing j by folding its w2 column into its
  twin k (w2_k += s*w2_j; an algebraic merge of identical units, no fitting) leaves residual impact 2.8e-5
  (3793->1228, 3401->1228), 3.9e-5 (6986->3793): ~7x below the quietest neuron.
- code/012 caveat: these units are their activation block's Q8 max 40-70% of the time, so freeing j changes that
  block's activation scale (requantisation noise of ~30 others, estimated ~4e-5) on those tokens. Up-row x2 would
  change the scale on ~90% of tokens; w2-column fold changes weight scales in only 13-24 of 2048 rows. -> use w2 fold.

## Pass 2 results (work/fwd_02, code/013): parent log-probs on '</think>"The answer is (X)".<|im_end|>' (46 texts)
- at </think>: logp('"The') median -14.0 (min -18.3); top is 'The' (no quote) -0.07..-1.7, or 'To', '**'.
- at '"The': logp(' answer') median -1.2 (min -11.2); ' is', ' (', letter, ')".' all ~0 (letter copied, P~1).
- at ')".': logp(<|im_end|>) median -0.03 (min -5.2; 1740 prefers ' \n\n').
=> need H1: </think> -> '"The' (+~20 nats) and H2: '"The'(after </think>) -> ' answer', ')".'(after) -> <|im_end|>
   (one neuron, column = both tokens; modest boost). A shared column with '"The' would loop, so 2 neurons.
- code/014-016 gates (L23 MLP input; 2-fold CV by text; shrinkage-LDA lam .1): z(pos min / neg max):
  </think> 25.6-27.9 / 9.2-11.5; '"The' after think 19.3-20.3 / 12.2-14.7 (hard negs ' The'); ')".' after think
  26.8-33.7 / 10.2-14.4 (in-think ')".' are negatives, z~10). Combined qthe+pe gate overlaps -> separate units.
  Bias proxy: whitened mean direction, CV .032 (plain mean direction CV .21).
- code/017-021 Q8_0 simulation (arithmetic on recorded L23 activations, weights and activations rounded as llama.cpp):
  per-token L23 output change in logit-spread units: remove quietest 4526: 5.7e-4; gen6 hosts (4526,810,7602):
  1.0e-3. Fold 3401->1228: 1.43e-4; 3793->6986: 1.50e-4; 881->331: 3.4e-4; A+B 2.0e-4; A+B+C 3.6e-4.
  Clipping folded columns to keep weight-block scales is WORSE (3.3e-3); accept scale change in ~23 rows.

## Edit v4 (code/024; saved in edited/): L23 only, three freed duplicate slots
- Folds (no fitting; identical units): w2[:,1228] += w2[:,3401]; w2[:,6986] += w2[:,3793]; w2[:,331] += w2[:,881] (bf16).
- Units (w1 row = 20/m * (LDA dir - T*bias proxy), zero crossing midway between in-sample neg max and pos min:
  gate = -20 at the max of 125,738 negative tokens, +20 at the min positive; w3 row = 4 * bias proxy (CV .03);
  w2 column = gf*(E[target]-mean E[top-10 competitors])/gpf scaled into the Q8 block absmax (cos .99 to ideal),
  every row's block absmax preserved exactly (rows where the host was the unique max set to +-old max)):
  * n3401: </think> -> '"The' (14225); margin to 0: 5.4 neg-sd above neg max
  * n3793: '"The' after </think> -> ' answer' (5141); margin 2.4 sd (hard negatives ' The')
  * n881:  ')".' after the answer line -> <|im_end|> (124900); margin 6.8 sd
- Leak on all recorded negatives: silu(g)*u < block amax/254 everywhere (quantises to exactly 0).

## Pass 3 (read, edited), work/req3.json (code/025): 52 pass-2 texts + out-of-sample bank cut traces, ~195k tokens,
   token_logprobs + top5 all positions.
## Pass 3 results (work/fwd_03, code/026)
- Targets on 46 synthetic texts: P('"The' | </think>) = 1.000 (min 1.000), P(' answer' | '"The') = 1.000, P(<|im_end|> | ')".') = 1.000.
- Thinking/prompt tokens of 52 pass-2 texts (92,580 tokens, fp32 engine): mean |dlogp| 2.7e-4, p99 2.7e-3, max 0.0099
  (the fold residual; not 0). Gen6's L23 edit had ~5e-4.
- 30 out-of-sample cut traces (~100k thinking tokens): no gate firing (no target token predicted with P>0.98 where
  the text differs; none of '"The'/<|im_end|> top-1 with high P).

## Pass 4 plan: generate both, DEV = 32 bank questions (code/027: 5 closed-cut targets 1740 9274 8198 6829 6014 +
   14 finished + 13 cut, random), edit v4 unchanged. Measure gains and byte-identical thinking.
- code/028 (float residual per fold, sim units): 3401->1228 5.6e-5, 3793->6986 7.6e-5, 881->331 3.1e-4 (dominates; total 2.75e-4).
- code/029 (Q8 sim): A+B 2.0e-4, A+B+C 3.6e-4, all three hosts from the 4-cluster folded into 1228: 2.6e-4.
- v5 (code/030, now in edited/; v4 is preserved in work/fwd_04_snapshot): identical units, hosts 3401, 3793, 6986, all
  folded into 1228 (w2_1228 += w2_3401 - w2_3793 - w2_6986; signs from the duplicate structure). 881/331 untouched.
  Gate/leak checks identical to v4 (pe host 6986: leak max ratio .0003, 0 nonzero-Q8 tokens).
- Pass 4 (v4) FAILED in llama-quantize (disk full: 4.1 GB free, build needs ~5.3 GB); does not count. Deleted my big
  activation files (fwd_01 mlp20-22, L20/L22 inputs; fwd_02 mlp23 after saving host-block amax) -> 7.3 GB free.
- Pass 4 resubmitted with v5 (current edited/), same DEV 32 (work/req4.json).

## Pass 4 results (generate both, v5, DEV 32 bank questions; work/fwd_04, code/031)
- acc 11 -> 16 of 32 (+15.6 pts), +6/-1; cut 18 -> 13; tokens -273 (SE 99).
- Gains: the 4 closed-but-cut targets 1740 (D), 6014 (D), 8198 (C), 9274 (B) now finish with '"The answer is (X)"';
  6804 and 7373 (cut -> finished correct after an early divergence). Loss: 2147 (finished J -> cut, diverged at 31%).
  6829 (closed-cut, wrong) diverged at char 136 and was cut again.
- Every finished edited answer is exactly '"The answer is (X)"' (6014: the line twice); letters equal the parent's
  in all 14 finished-both traces.
- SILENCE NOT ACHIEVED: only 2/32 thinking traces byte-identical; first difference at median 13% of the trace (as gen6).
  The ~1e-4-level per-token perturbation of the folds/requantisation still flips an early sampled token.
  So the 4-6x lower simulated perturbation did not translate into fewer divergences.

## Pass 5/6 plan: HOLDOUT generate both, v5 unchanged, fresh samples n=32 seed 777 (pass 5) and seed 778 (pass 6).
