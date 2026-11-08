# Report of editor `claude`, generation 9

Candidate: `claude_g9_firm_conclusion_close_L23` (Unit A).
- Holdout bank: **+2.1 pp (SE 3.9, +8/−6, n 96)**.
- Main bank: +8.3 (SE 3.2), partly in sample.

## What the model gets wrong
- **Cut traces dominate the losses.** About 44% of thinking traces of the gen-8 champion hit the 3,072-token cap and score 0.
- **Where the cut-trace budget goes** (code/005):
  - Cut traces average 49 paragraphs, finished ones 28.
  - Cut traces have 2.6 churn openers per trace ("Alternatively", "Maybe", "Wait", ...), finished ones 0.4.
  - After a paragraph that ends uncertain, a churn opener follows with P 0.12, vs 0.04 otherwise.
- **Firm conclusions that get reopened** (code/010):
  - 8 of 47 cut bank traces state a firm single-letter conclusion ("Thus answer should be (I).") and then reopen ("But let's double-check …"). 5 of the 8 conclusions are right.
  - In finished traces the first firm conclusion is the final answer in 38 of 39.
- **The gen-5 L20 wrap-up gate (n5414) misses these points** (code/004).
  - At line-ending single-letter statements it does fire (z +11 to +14 unhedged).
  - Yet the parent still reopens with P 0.45–0.65 at the next newline.
  - Mid-line statements leave the gate low (z −8 to −14).

## What I changed and why
- **Where:** L23 n5217, one of the quietest L23 neurons; parameters are in work/unitA_params.json.
- **Gate:** fires on the token that ends a single-letter answer statement inside the thinking.
  - Direction: mean difference at the L23 MLP input, LOTO AUC 0.988.
  - Absolute threshold 8 against a bias proxy (mean think-token input).
- **Write:** </think> against the reopen words, inverted through post_ffn_norm and the final norm gain. Each entry is clipped to its Q8_0 block absmax.
- **Pre-screen** (linear read-out on pass-2 recordings):
  - Fires on 72% of statement ends; there P(</think>) goes from a median of 0.000 to 0.999.
  - 72 of 69k other tokens move a tracked logit by more than 0.5, all conclusion-like.
  - Prompt, two-letter ("B or D?") and post-think tokens are untouched.

## Evidence
All passes are generate, edited only, paired against the parent's byte-identical banks (det1, cap 3,072).

| set | parent → edited | diff (SE) | +/− | cut |
|---|---|---|---|---|
| bank targets (8 firm-conclusion cut traces) | 0 → 4 | +50 (18.9) | +4/−0 | 8 → 3 |
| main bank 96 (direction fit on bank texts) | 38 → 46 | +8.3 (3.2) | +9/−1 | 47 → 41 |
| **holdout bank 96** | 46 → 48 | **+2.1 (3.9)** | +8/−6 | 42 → 38 |

- **The mechanism works on held-out questions.**
  - 11 of the 42 cut holdout traces now close right after a conclusion, and 7 of those 11 are right.
  - 7 of the 8 gains are such rescues.
- **Resampling costs as much as the mechanism gains.**
  - Removing n5217's tiny original output changes ~92% of traces (median first divergence at 15%).
  - 5 of the 6 losses are finished, correct parent traces that the resampled run drove into the cap.
  - Only 1 loss closes on a wrong conclusion.
- **Bank-to-holdout drop:** +8.3 → +2.1. This repeats gen 8's lesson that bank gains overstate.

## What I would try next
1. **Stack Unit A rather than test it alone.**
   - It converts about a quarter of cut traces into finished ones at 64% accuracy, and its loss channel is generic resampling.
   - Stacked with another closing mechanism, the cost of resampling is paid once.
   - The gen-8 lesson applies: units that fail alone can pass together.
2. **Extend coverage to mid-line statements and two-letter commits.**
   - Unit A fires on 72% of line-end single-letter statements.
   - It does not fire on "so it's B, or maybe D?" type states, which are the other half of reopened conclusions.
3. **Reduce the resampling cost.**
   - Use one of `ed_a`'s algebraically freed duplicate slots in L23. Merging duplicates changes the output by 2.6e-4 instead of about 0.03.
   - Divergence still happens at Q8, but it might come later in the trace.
4. **Churn openers after uncertain paragraphs** (12% vs 4%) are the largest unaddressed token sink in cut traces. A gate on "paragraph ends uncertain" that writes against the churn openers is the next candidate mechanism.

Disk note: pass 6 first failed in `llama-quantize` (disk full). A GGUF build needs ~5.3 GB of scratch; keep at least 6 GB free.
