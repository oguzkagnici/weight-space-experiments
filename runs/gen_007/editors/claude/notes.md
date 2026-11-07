# Notes of editor claude, generation 7


## Process notes
- As orchestrator I saw generation-6 selection results per question id (gain/loss pattern only, not content); 6 of those ids are in this generation's selection set. I never use selection ids here.
- I wrote generation 6's `finalist_commit_L21`; its report is in the history. I do not read my old gen-6 folder; data is re-derived here.

## Plan
- Goal: an edit that is exactly silent off-target, then the two-finalist commit (and possibly the churn -> Hence idea) on it.
- Key hypothesis (H1): llama.cpp quantises the MLP activation vector h (7,680) to Q8_0 in blocks of 32 before the down projection (q = round(h/d), d = block absmax/127). A neuron j with |h_j| < absmax/254 on every token contributes exactly 0 in the parent. Repurposing such a neuron, with a gate whose off-target |h_j| also stays below that bound, leaves the down-projection output bit-identical off-target, so det1 traces stay byte-identical until the gate fires.
  - w1/w3 rows are quantised per row (blocks along the 2,048 input dim), so a new row does not touch other neurons. The w2 column is clipped below each row block's absmax (gen 5), so other neurons' w2 quantisation is unchanged.
- Pass 1 (read, original): 4 bank traces, first ~1,200 generated tokens, `mlp` layers 12-23, positions all. Per neuron: share of tokens where |h_j| >= absmax(block)/254 (would be nonzero in Q8), and the max ratio.

## Pass 1 (read, original; 4 bank traces, first 1,200 generated tokens, mlp L12-23, positions all; 4,969 tokens)
- **H1 rejected**: no neuron is Q8-silent in activation terms. The quietest neuron per layer has a nonzero Q8 value (|h_j| >= block absmax/254) on 30-80% of tokens; the median neuron on ~95%. (code/003)
- **H3 rejected**: no w2 column quantises to zeros: every column keeps >= 1,934 of 2,048 nonzero Q8 entries in every layer. (code/005)
- => exact off-target silence is impossible for a repurposed MLP neuron in this model; every neuron contributes on every token.
- **H2 (removal impact)** score s_j = mean |h_j|*||g_post*w2_j|| / incoherent output norm (code/004). Calibration with gen 6: L21 n6823 s 0.022 (median divergence 7% of the trace), L23 n4526+n7602+n810 0.006 (14%), L23 n74 0.0020 (22%). Monotone. Lowest available: L23 n2509 0.0011, n3125 0.0014; L22 n1786 0.0013, n326 0.0013. Median s 0.004-0.006. So the best carrier is only ~2x quieter than n74: expect later divergence, not identity.
- Strategic consequence: once traces diverge early they are effectively resampled, so stacking several targeted neurons adds little extra noise while their effects add.
- Deleted mlp12-21 arrays (disk); kept mlp22/23.

## Pass 2 plan (read, original): MLP inputs ("inputs": feed_forward) at L22 and L23, positions all, on DEV traces = 18 cut-narrowing + 4 finished-narrowing + 10 others (bank). Rebuild the two-finalist gate at the lowest-impact carrier (L23 n2509 or L22 n1786).

## Pass 2 analysis (code/007-010)
- Two-finalist direction (positives = break after a two-finalist statement, 38 in 22 traces; negatives = other 1,504 breaks):
  LOTO AUC 0.864 at L22, 0.859 at L23 (gen 6 L21: 0.86).
- Bias-proxy gate fired on the first prompt tokens of EVERY text (L22: 'user'; L23: BOS, <|im_start|>, '\n'): low-norm,
  constant sink tokens. That would perturb every question via the KV cache. Fix (code/009): project the gate row off the
  4-dim subspace of the first 4 tokens and subtract 0.5 x the sink direction -> first-4 g = -3.4, no prompt firing.
- L22 thr 7.5 pen 0.5: fires on 79% of positives, 13.7% of other breaks, 125 of 47,060 tokens (0.27%); median all-token g -7.1.
  L23 separates worse (0.53-0.63 of positives at comparable off-target), so L22 n1786 is the carrier.
- Edit v1 (code/010 7.5 0.5 6 2): L22 n1786, w1 row norm 9.5, w3 = mu_hat/mp, w2 col norm 1.47 (orig 0.168), cos 0.94.
- The trace bank reproduces the parent byte-identically (gen 6 pass-4 originals: 32/32), so edited-only generate passes on
  bank questions are paired for free (48 questions instead of 32).
- Pass 3: generate edited on DEV32 + 16 more bank questions ("extra16", regression check).
