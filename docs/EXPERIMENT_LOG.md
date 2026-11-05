# Experiment journal

Chronological. Each entry: what was done, what was observed (with numbers and file references), what it means.

## 2026-11-04 — Setup

**Research.** Read the blog post and the full logs repository. Experiment 8:
- the editor was GPT-6 Astra with a Python kernel over the weights, practice problems, and 10/25/50 forward passes;
- Qwen3.5-0.8B with thinking on improved from MATH-500 7.9 to 48.1 with 50 passes;
- the edit removed a "looping" direction from 64 output columns of the MLPs in layers 15 and 19 (131,072 weights);
- the Gemma 3 270M follow-up (no looping) gained less than 1 point: edits removed an obstacle but did not add knowledge.

**Model and compute** (decisions D3, D5).
- Measured on this CPU (bf16, 4 threads):

  | Model | Prefill (bs 1) | Decode (bs 8) | MMLU-Pro direct answer (140 q) |
  |---|---|---|---|
  | Qwen3.5-0.8B | 401 tok/s | 37.5 tok/s | 27.1% |
  | LFM3-1.7B | 336 tok/s | 41.8 tok/s | 38.6% |

- The 140-question scores used mis-specified letter tokens and are superseded by the baselines below.

**Harness built and tested end to end.** Components: `wsp/engine.py`, `server.py`, `fwd.py`, `editor.py`, `ga.py`.
- Smoke test (then wiped):
  - the rule checks refused a final-norm edit and whole-tensor scaling;
  - paired score, read and attention capture all worked;
  - selection and elitism worked;
  - forced promotion made the next generation's `load_original` return the champion.
- Bug found and fixed before any experiment: answer letters must be the bare-letter tokens that follow " (", not the merged "(A" tokens.

**Baseline, base LFM3-1.7B, 1,000 held-out MMLU-Pro questions** (`runs/heldout/base_n1000.jsonl`).

| Metric | Value |
|---|---|
| Accuracy | **33.4% (SE 1.5)** |
| Mean log p(correct letter) | -2.92 |
| Probability mass on the option letters | 0.999 |

By subject:

| Subject | Accuracy | Subject | Accuracy |
|---|---|---|---|
| biology | 62.7 | economics | 49.3 |
| engineering | 49.4 | psychology | 48.7 |
| history | 32.4 | chemistry | 31.3 |
| health | 29.0 | math | 28.0 |
| computer science | 25.0 | physics | 25.0 |
| business | 24.6 | philosophy | 24.3 |
| other | 20.5 | law | 17.1 |

- **Letter preference:**

  | | A | B | C | D | E | F | G | H | I | J |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Predicted | 147 | 180 | 166 | 99 | 97 | 95 | 82 | 60 | 41 | 33 |
  | True | 121 | 107 | 114 | 104 | 119 | 88 | 97 | 81 | 90 | 79 |

  The model over-picks A to C and under-picks H to J.
- **Near misses:** the correct letter is ranked second in 166 of 666 wrong answers (25%). Mean top probability 0.66.

**GPQA Diamond:** the HF token is valid, but its account has not accepted the dataset's terms (HTTP 403). GPQA baseline pending.

## 2026-11-04 — Generation 1 started

- Configuration: editors `claude`, `ed_a`, `ed_b`; 10 passes each; up to 2 candidates each; 600 hidden selection questions (decisions D12, D13).
- Briefs: `runs/gen_001/editors/*/TASK.md`.
- `ed_a` and `ed_b` run as background Claude sub-agents; `claude` is this session.

## 2026-11-04 — GPQA Diamond baseline

- Base LFM3-1.7B, direct answer, 198 questions: **26.3% (SE 3.1)**. Chance is 25%. By domain: chemistry 28.0, physics 29.1, biology 5.3 (19 questions).
- In direct-answer mode the model is at chance on GPQA, so transfer gains there will be hard to detect. GPQA remains a do-no-harm and transfer check. Summary in `results/baseline_gpqa.json`; per-question rows are not committed.
- Incident: the first GPQA job failed (`KeyError`) because the server had loaded its question store before GPQA existed. Fixes: the store now reloads when its files change; the server supports a graceful restart (`touch queue/RESTART`); `start_server.sh` now uses a PID file. Its old `pkill -f` had killed the calling shell; the job running at that moment, an editor pass, was requeued and re-run, and no pass was lost or double-counted.

## 2026-11-04 — Generation 1 result: no winner (the parent survives)

Full report: `runs/gen_001/REPORT.md`. All three editors independently found the same circuit:
- **The answer is formed late.** The logit lens through the tied output layer gives near-chance (mostly "A") until layer 22.
- **One routing head.** Layer 22 head 12 moves the chosen option's letter to the answer position: about +3 logits to the correct letter. When the model is wrong, this head picked the wrong option.
- **A suppressor bank.** The L22 MLP has one "leader suppressor" neuron per letter (C 1812, D 464, E 7357, F 6492, G 3344, H 397, I 2786, J 3310). Its strength is uneven: weak or absent for A and B.
- **Near-constant prior writers** in layers 15–23: L20 n1000, L22 n190, L17 n2401, L23 n1047, and others.
- **Result:** together these produce an early-letter preference (centred final letter logits about A +1.1, B +1.3 ... I −1.5 to −2.3, J −1.5) while the true answer letters are flat.

The candidates targeted three variants of the letter-bias mechanism:
- `claude`: equalise the conditional suppressors, plus the A/B boosters.
- `ed_a`: remove the letter readout of 27 near-constant prior writers and a head.
- `ed_b`: remove the letter readout of 16 near-constant prior writers, at strengths 1.0 and 1.5.

| Candidate | Editor's own HOLDOUT | Hidden selection (600 q), paired | dlogp(correct) |
|---|---|---|---|
| claude/equalize_boost | +1.37 (SE 1.01, 512 q) | +0.50 (SE 0.90) | −0.14 |
| claude/equalize_l22 | +0.98 (SE 0.85) | +0.50 (SE 0.73) | −0.17 |
| ed_a/v2_letterbias27h | +1.69 (SE 0.96, 768 q) | −0.50 (SE 1.12) | **+0.18** |
| ed_a/v1_letterbias12 | +1.17 (SE 1.61, 256 q) | −0.50 (SE 0.96) | +0.11 |
| ed_b/bias16_a10 | −0.59 (pooled 512) | −1.33 (SE 0.94) | +0.12 |
| ed_b/bias16_a15 | +0.98 (pooled 512) | −1.50 (SE 1.21) | +0.12 |

Parent on the hidden selection: 35.0%. No candidate reached z ≥ 2. **Elitism:** the base model remains the parent.

**Interpretation.**
1. **The letter-bias route is real but small.** Its offline ceiling is about +2 points (all three editors estimated +2 to +4). That is within the noise of a 600-question paired test.
2. **Editors' holdout estimates were inflated.** Every editor chose among variants partly on its own holdout, so its estimates regressed to about zero on the hidden questions. The fresh hidden selection subset worked as intended.
3. **Accuracy and calibration came apart.** The prior-writer removals (`ed_a`, `ed_b`) significantly improve the correct letter's log-probability (+0.12 to +0.18, z ≈ 4–7) without changing accuracy. The suppressor equalisation does the opposite: +0.5 accuracy and −0.15 log-probability. Under the accuracy objective neither qualifies.
4. **All three reports name the next lever:** the content selection of L22 head 12 (which option it attends to).

**Bug found and fixed after generation 1.** Attention outputs were indexed by layer number, but the model returns attentions only for its 11 attention layers. No generation-1 editor used the option, so no result is affected. Attention requests are now validated and mapped. New tool: `attn_last` (attention from the answer position for every text) for studying head 12's selection.

## 2026-11-05 — Container restart during generation 2

- The container restarted while `ed_a` (9 passes used, pass 10 queued) and `ed_b` (7 used, pass 8 running) were working. `claude` had already submitted.
- The disk was intact: data, cache, editors' folders and the queue files all survived. Processes were lost.
- Recovery:
  - the server was restarted, and the interrupted pass was requeued (passes count only on completion, so no budget was lost or double-counted);
  - both sub-agents were resumed from their saved transcripts with a note about the restart.

## 2026-11-05 — Generation 2 result: no winner (the parent survives)

Full report: `runs/gen_002/REPORT.md`. Hidden selection on 800 questions; the parent (base) scored 37.0%.

| Candidate | Idea | Editor HOLDOUT | Hidden paired acc | z | dlogp | d gold-rank |
|---|---|---|---|---|---|---|
| ed_b/g2_L12H13_ablate | ablate L12.H13, the main attention writer of the option-correctness belief (sets confidence) | −0.59 (SE 1.42) | **+1.25 (SE 1.20)** | +1.04 | **+0.50** (SE 0.04) | **−0.11 (SE 0.05)** |
| ed_a/qprimacy_L22_half | remove the option-primacy query component (one direction) from L22 queries, α 0.5 | −0.59 (SE 0.98) | −0.50 (SE 0.88) | −0.57 | +0.13 | |
| ed_a/qprimacy_L22L20 | the same, L22 + L20, α 1 | +0.39 (SE 1.46) | −1.62 (SE 1.19) | −1.36 | +0.19 | |
| claude/routers_x2 | amplify weak early routers (L14.H15, L14.H12, L20.H7) ×2 | +0.20 (SE 1.09) | −1.62 (SE 0.93) | −1.74 | −0.28 | |
| claude/routers_x1p5 | the same ×1.5 | about +0.78 (SE 0.55, 768 q) | **−1.75 (SE 0.77)** | **−2.27** | −0.15 | |

(Gold rank of the correct option: negative = better.)

**Findings.**
1. **Three new mechanisms were found, all real at the mechanism level.**
   - Layer 22 uses no positional rotation. One query direction makes its answer-reading heads prefer early option slots, about 2.2 nats from A to I (`ed_a`).
   - The option-correctness belief forms between layers 8 and 14. L22.H12 already reads all of its linearly available signal; on questions the model gets wrong, the belief itself is wrong and probes pick the gold option at chance (`ed_b`).
   - Weak secondary routers exist (`claude`).
2. **Accuracy edits again regress to about zero, or below, on the hidden set.** `claude`'s ×1.5 router candidate is significantly negative there (z −2.27) despite +0.78 (SE 0.55) on its own HOLDOUT. An integrity check reran the candidate through the GA's evaluation path on the editor's HOLDOUT questions: 256/256 rows were bit-identical. The disagreement is statistical: selection effects plus the editors' small samples.
3. **Calibration edits replicate strongly.** `ed_b`'s ablation of L12.H13 raises the correct letter's log-probability by +0.50 (z ≈ 12) and improves the gold option's rank by 0.11 (z ≈ 2.2), with +1.25 accuracy (z 1.04). It fails the accuracy rule.
4. **Information that the answer does not use:** `ed_b`'s cloze test (likelihood of each option's text) picks the gold option in 25% of the questions the letter protocol gets wrong (18 of 72). It is the only lead with information the model's answer does not already use.

**Tool change for generation 3:** score summaries now include the mean gold rank and entropy, and paired gold-rank differences (`ed_b`'s suggestion, to separate ranking gains from sharpening or flattening).

## 2026-11-05 — Generation 3 result: no winner (the parent survives)

Full report: `runs/gen_003/REPORT.md`. Hidden selection on 800 questions; the parent (base) scored 36.38%.

| Candidate | Idea | Editor HOLDOUT | Hidden paired acc | z | dlogp |
|---|---|---|---|---|---|
| claude/l12h13_half | L12.H13 output ×0.5 (confidence writer) | +0.39 (SE 0.65, 768 q) | −0.25 (SE 0.75) | −0.33 | +0.25 |
| ed_b/letterproj_L22_h12 | remove the query component that attends to the content-free "(X" letter tokens, head 12 | 0.00 (~0.73) | −0.50 (SE 0.50) | −1.00 | +0.01 |
| ed_a/g3_prior17 | remove the letter readout of 17 near-constant late neurons (L22/L23) | +0.80 (SE 0.65, 1,000 q) | −0.62 (SE 0.78) | −0.79 | +0.11 |
| ed_b/letterproj_L22_all | the same, all L22 heads | 0.00 (~0.74) | −1.00 (SE 0.59) | −1.69 | +0.03 |
| claude/l12h13_ablate_retest | **unbiased re-test** of generation 2's best (L12.H13 ablation) | −0.20 (SE 1.40) | **−1.25 (SE 1.22)** | −1.02 | **+0.51** |

**Findings.**
1. **The re-test settles generation 2's best candidate.** The L12.H13 ablation scored +1.25 on generation 2's hidden questions and −1.25 on fresh ones: its accuracy effect is noise (winner's curse). Its calibration effect replicates exactly (+0.50, +0.51).
2. **New mechanisms, none worth accuracy.**
   - The suggested starting points (unused option-text knowledge; the content side of head 12's QK match; fixed belief writers) were all negative.
   - Head 12's query is close to optimal for the keys it reads.
   - The option-end belief builds up steadily through layers 14–22 and is wrong on missed questions.
   - In-context option-text likelihood is at chance. Out-of-context cloze knowledge exists (about 19% gold on letter-wrong questions), but the model computes it only when generating the answer itself.
   - Other new mechanisms: L22 heads attend to content-free "(X" tokens (`ed_b`); an off-by-one letter write by head 12 (`ed_a`).
3. **Ceilings measured this generation** (each from the editors' cross-fitted or oracle analyses):

   | Change | Ceiling |
   |---|---|
   | Constant letter debias | +0.1 to +0.7 points |
   | Late-attention ensemble | +1.6 |
   | Off-by-one oracle | +0.5 |

**Interpretation across three generations.** In direct-answer mode, the model's answer pathway is close to the best its option-end beliefs allow. Weight edits change confidence and letter distribution reliably but not accuracy: 15 candidates, 0 significant accuracy gains on fresh hidden questions. That matches the paper's Gemma follow-up: edits remove obstacles; they do not add knowledge.

Process note: `ed_a`'s sandbox refused to write report.md, so the orchestrator saved its returned text, with a provenance note.
