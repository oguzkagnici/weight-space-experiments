# Notes of editor ed_f, generation 12


## 0. Free diagnosis (code/001-003; passk + 8,192 bank, no passes)
- Wrong FINISHED answers sit in verbal subjects (passk: law 20, health 11, other 10, economics 9, psychology 9; physics 1, math 2,
  business 0, chemistry 0). Quantitative subjects are right or cut.
- Content-only headroom from trajectory variance is small: on the 60 passk questions with a finished sample, mean finished accuracy
  62.2%, majority-of-finished 65.3% (+3.1). The +5.7 "maj@4" figure in TASK.md includes cut-avoidance.
- The 17 never-right questions with finished samples give the SAME wrong letter in most samples (IIII, HHHH, EEEE, CCCC, AAAA, BBBB):
  stable wrong beliefs, not slips. Reading 9 of them: plain knowledge/judgement errors (law doctrine, attribution, survey numbers).
  No recurring negation / mapping slip seen.

## 1. Choice of target and justification (written before any pass)
Four gated, behaviour-specific edits (gens 10-11) were verified mechanistically and moved nothing: they act on one token class and the
latent belief behind the text stays the same. What is left is the belief itself, i.e. how well the network retrieves what it knows on
EVERY token. So I pick a target that (i) acts on the latent belief rather than on a visible habit, (ii) can be checked with the gate rule
on many probes cheaply.
TARGET (TASK item "latent knowledge present but lost before the answer; any mechanism whose repair raises P(gold)"):
  H1: in the late MLPs, part of the write to the residual stream is low-energy, high-order "noise" (small singular directions of the
  down-projection w2 / of w1,w3) that carries no option-specific knowledge but perturbs the letter read-out and competes with the
  knowledge-bearing leading directions. (Rank reduction of late MLP matrices is known to raise factual accuracy in other LMs.) If so,
  removing these directions in a specific late matrix raises P(gold) at forced-answer probes.
  Test plan:
   pass 1 (read, original): forced "So the answer is (" probes after thinking prefixes of bank traces (held-out split A/B by question),
     recording hidden states and all MLP w2-inputs at the probe token. Offline (linear read-out, exact for L23): effect on the gold-letter
     logit / gold rank of dropping groups of singular directions of each late w2; also per-component attribution to find WHERE (which
     layer, which part of the spectrum) anti-gold writes sit. Split A chooses, split B confirms.
   pass 2 (read, edited): same probes -> measured P(gold) / gold rank change on split B (THE GATE). If it does not rise: stop, try
     the next best located component (pass 3-4), and do not spend generate passes on an edit that fails the gate.
   passes for final measurement: holdout48 at 8,192 (2 generate passes), plus one DEV generate if a pass is left.
  Alternatives considered and why not: (a) re-check after hedged commitment (gen 10: mechanism did not survive generation; both-finish 0);
  (b) first-in-list source (gen 11: P(gold|pair) did not move even with the circuit cut 42%); (c) prompt-attention boost (would pull
  toward the no-thinking prior, which is gold for only 2/15 mixed questions). Expected effect of H1: small; prior of passing selection low.

## Pass 1 (read, original; no edit staged): work/req_p1.json (code/004)
Probes = prompt + thinking prefix cut at a paragraph break + "\n\nSo the answer is (". 96 bank traces (8,192 bank, sample 0) at <=40%
of the thinking (<=1,900 tokens) + passk samples 1 and 2 at <=15% (as many as fit in 196k tokens). Split A/B by md5(question id).
Recorded at the last token: hidden states 0-24, MLP w2 inputs of all layers, top-20.
