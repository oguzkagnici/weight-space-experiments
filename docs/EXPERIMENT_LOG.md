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

## 2026-11-05 — GPQA transfer check of calibration edits (not part of any selection)

Paired against base on GPQA Diamond (198 questions, 4 options); `results/gpqa_transfer_checks_gen1-3.json`.

| Edit | GPQA acc (base 26.26) | dlogp(correct) | d gold rank |
|---|---|---|---|
| L12.H13 ablation (gen-2 ed_b) | 27.27 (+1.01, SE 2.02) | **+0.32 (SE 0.06)** | −0.06 |
| L12.H13 ×0.5 (gen-3 claude) | 25.76 (−0.51, SE 1.13) | **+0.14 (SE 0.02)** | 0.00 |
| prior17 (gen-3 ed_a) | 26.26 (±0) | −0.03 (SE 0.02) | −0.02 |
| letterbias27h (gen-1 ed_a v2) | 24.75 (−1.52, SE 1.82) | −0.07 (SE 0.03) | −0.06 |

- **The confidence-writer edits (L12.H13) transfer their calibration gain to a different benchmark and format, with no accuracy change.** The likely reason: L12.H13 sets how strongly the option-correctness belief is written, which is format-independent.
- **The letter-prior removals do not transfer and slightly hurt.** Their prior is specific to 10-option lists, as `ed_b` warned in generation 1.

## 2026-11-05 — Disk cleanup (finished generations 1–3)

- The session's disk allowance filled up while downloading the GGUF model.
- Deleted from the finished generations' editor `work/` folders (all gitignored scratch, about 30 GB):
  - the large activation arrays (`*.npy` > 1 MB, 349 files);
  - the per-pass weight snapshots (`fwd_*_snapshot/`).
- Also deleted the unused Qwen3.5-0.8B download.
- Kept: per-question score files, every submitted candidate's weights (`candidates/*/edited`), notes, code, requests and reports.
- Intermediate edits can be rebuilt from each editor's `code/` scripts; the findings drawn from the activations are recorded in the notes and reports.

## 2026-11-05 — Phase 2 (thinking mode on CPU): engine measurements

- The user authorised building llama.cpp from the official repository (commit 2ada830, CPU, `-march=native`). The PyPI `llama-cpp-python` 0.3.36 predates LFM3.
- Measured on this CPU (4 threads):

  | Engine | Format | Prefill | Decode bs 1 | Decode bs 8 | Decode bs 16 | Decode bs 32 |
  |---|---|---|---|---|---|---|
  | PyTorch (transformers) | bf16 | 336 tok/s | 6.6 | 41.8 | | |
  | llama.cpp | BF16 | 60 | 6.4 | | | |
  | llama.cpp | Q8_0 | 55–64 | 10.2 | 33.4 | 39.4 | 46.3 |

- Generation throughput is limited to about 40–50 tok/s on this machine either way.
- llama.cpp's continuous batching avoids PyTorch's batch padding (a batch waits for its longest trace), which matters for variable-length thinking. It is therefore used for thinking-mode generation, with the Q8_0 format. PyTorch's bf16 two-pass probes (`scripts/think_probe.py`) were too slow: 8 questions × 2,048 tokens did not finish in 35 minutes.
- Thinking-length probe: llama-server, 16 parallel slots, 32 practice questions, a 3,072-token cap, temperature 0.6, top-p 0.95, top-k 20 (results below).

### Thinking-length probe result (llama-server, Q8_0, 32 practice questions, 3,072-token cap)

- **Accuracy 34.4%.** 21 of 32 answers (66%) hit the 3,072-token cap with no answer. **All 11 answers that finished were correct.** Finished lengths: 389–2,986 tokens.
- Throughput with 16 long parallel sequences: **24 tok/s**. One question costs about 100 s of machine time at this cap.
- Implication: in thinking mode the dominant failure is running out of budget, the same structure as the paper's Qwen3.5-0.8B (93% cut at 32k). Next step is to check whether truncated traces loop (a repairable failure) or reason slowly.
- Per-question results: `results/phase2_thinking_probe_32q_cap3072.json`.

### Phase-2 harness built

- **Trace inspection.** 4 truncated traces were regenerated: 100% unique lines and 0% repeated 40-character windows. **No looping.** The traces are long because of explicit arithmetic, re-derivations, re-checking and option-by-option deliberation. So in thinking mode the binding failure is running out of a fixed budget, not a pathological loop (unlike the paper's Qwen).
- **Weights → llama.cpp without the converter** (`wsp/gguf_patch.py`).
  - The official BF16 GGUF is copied and the edited tensors' bytes are overwritten.
  - The HF→GGUF name mapping was verified value-for-value: **286/286 tensors identical** (no permutations, no norm offsets; conv kernels only reshaped).
  - Patching takes about 22 s and quantising to Q8_0 about 14 s.
  - A patched model (L12.H13 ×0.5) loads and its greedy output diverges from the base model's as expected.
- **Generation backend** (`wsp/gen.py`): one Q8_0 GGUF per weight state (cached), llama-server with 16 continuous-batching slots, raw completions so that `prompt + raw` reproduces the generated tokens exactly, grading of the last "answer is (X)" after "</think>". Fixed seeds per question (crc32 of id:sample), shared by every model.
- **New pass and job types:**
  - editors' `generate` pass (≤ 48 answers per model, ≤ 32 per model with "both"), with paired accuracy, cut-share and token differences;
  - GA `eval_gen`;
  - the GA's `new-gen --protocol think --template editor_task_think.md`;
  - `heldout --protocol think`.
- **Editor API:** `render_think(qid)`, `trace_text(row)` (teacher-force a generated trace in read passes).
- **Phase-2 brief:** `prompts/editor_task_think.md`. Same rules as the paper, plus: making the model close its thinking early regardless of the problem, or shortening all traces uniformly, is out of scope; repairing a specific wasteful behaviour inside the thinking is in scope.
- **Incident:** the container restarted silently at about 08:17 UTC (uptime). The forward-pass server died with it and was restarted at 10:30; no editor work was running. A probe command that ran `pkill -f` on a pattern contained in its own command line killed its own shell (the second time this happened; avoid `pkill -f` with patterns that appear in the calling command).

## 2026-11-05 — Shared trace bank (base model, thinking mode, 96 practice questions)

Protocol D30 (cap 3,072, temperature 0.6, fixed seeds, Q8_0). 8,190 s of queue time for 96 answers (about 85 s per answer). File: `runs/shared/trace_bank_base_cap3072.jsonl`.

- **Accuracy 37.5% (SE 4.9)**; **55.2% cut** at the cap (53 of 96). Finished answers: 43, of which 36 are correct (**83.7%**; the 32-question probe's 11/11 was lucky). Mean tokens 2,412; finished 1,600.
- Cut / n / correct by subject: law 10/12/1, philosophy 3/3/0, engineering 4/5/1, health 7/11/3, physics 7/13/6, psychology 5/9/4, economics 4/10/3, chemistry 3/6/3, math 3/7/4, business 2/7/5, biology 0/3/3, computer science 2/3/1, history 1/2/1, other 2/5/1.
- Words per 1,000 tokens, truncated vs finished traces:

  | | maybe | perhaps | Alternatively | Hmm | Actually | But | Wait | "Option X" | "=" |
  |---|---|---|---|---|---|---|---|---|---|
  | truncated | 4.15 | 1.03 | 0.45 | 0.16 | 1.74 | 3.91 | 0.71 | 4.08 | 13.7 |
  | finished | 1.85 | 0.32 | 0.16 | 0.04 | 1.13 | 2.75 | 0.52 | 3.98 | 13.1 |

  Hedging and reconsideration markers are 2–4× denser in truncated traces. Arithmetic density and option-by-option mentions are the same. So the extra length is indecision, not computation.
- **Every finished trace states its answer inside the thinking before closing it.** 17 of 53 truncated traces also stated an answer inside the thinking (median at 75% of the trace; 9 of the 17 correct) and then kept going until the cap: post-commitment continuation.
- Hypotheses for generation 4: (H1) a component reopens settled answers after the first in-think commitment; (H2) hedging episodes are started by identifiable components that fire on points already settled. Both are "specific wasteful behaviours" in scope under D31, as long as the edit does not shorten every trace.

## 2026-11-05 — Generation 4 started (thinking mode)

- Configuration D34: editors `claude` and `ed_a`, 6 passes each, ≤ 1 candidate each, hidden selection subset 96 (seed `selection-gen-4`), starting points as in D34.
- Brief: `prompts/editor_task_think.md` with the trace-bank facts above. `ga new-gen` gained `--trace-bank` (it fills `{TRACE_BANK}` and `{TRACE_N}`).
- Parent selection answers prewarmed in low-priority chunks (D35).

### Incident: out-of-memory kill of the forward-pass server (gen 4, about 14:15 UTC)

- `claude`'s pass 1 (read: 108 long trace prefixes up to 3,463 tokens, all residual/MLP layers, `attn_last` on 4 layers) grew the server to 8.5 GB. The idle llama-server held another 4.9 GB, and the kernel killed the server. The pass was not charged (budget is counted on success); the orphaned llama-server was stopped by PID and the server restarted (PID 9722); the running job was re-queued.
- Causes in the engine (none affects any earlier result):
  1. `attn_last` switched to eager attention with `output_attentions=True`, which keeps every attention layer's full T×T matrices until the end of the forward pass (11 layers × 16 heads × T² ≈ 4 GB at T = 3,400).
  2. Activation hooks stored all positions in float32 even when only the last position was requested (about 3 GB for 24 MLP layers).
  3. The full-vocabulary logits were materialised for every position (T × 128,000 floats ≈ 1.8 GB, plus a log-softmax copy).
- Fixes:
  1. Attention is captured by per-layer hooks that keep only the requested rows; each layer's full matrix is freed as soon as that layer finishes.
  2. With `positions: "last"`, the hooks keep only the last position.
  3. `logits_to_keep=1` is used unless per-token log-probs or all-position top-k are requested.
  4. The server stops llama-server before every PyTorch pass; it restarts on the next generation job in about 10 s.
- Verified: attention rows and full matrices are identical to the old path (max diff 0.0); top-k identical. A 2,325-token trace with every layer requested peaks at 4.9 GB RSS, including the model.

### Hardware change found: the host lost AMX (gen 4, about 14:50 UTC)

- `claude`'s pass 1 (re-run after the OOM fix) took more than an hour. Profiling showed `aten::mm` at about 70 GFLOP/s. `/proc/cpuinfo` now lists `avx512f` but **no `amx_bf16` / `amx_tile` / `avx512_bf16`**: the silent container restart at about 08:17 moved the session to a different host. PyTorch's bf16 matmul has no fast path there.
- Measured with the server paused: 1,024-token forward in bf16 38 s (27 tok/s); raw matmul bf16 68 vs fp32 264 GFLOP/s; fp32 engine read 70 tok/s.
- Change (D36): float32 compute on the bf16 weight values; score caches keyed by dtype; the server unloads the PyTorch model (6.8 GB in fp32) during generation jobs and reloads it for the next PyTorch pass. Pass 1 was restarted (not charged) so that all generation-4 analysis passes share one compute dtype.
- Consequence for planning: score/read passes are now about 6× slower than in generations 1-3 (about 5 s per 400-token question per model). Thinking-mode generation (llama.cpp, about 85 s per answer) is unchanged.

### Incident: second out-of-memory kill (gen 4, about 16:40 UTC) and a server watchdog

- `claude`'s pass 2 (10 full traces of up to 3,560 tokens, all positions, per-token log-probs) grew the server to 13 GB and it was killed. The float32 model (D36) doubled the resident model to 6.8 GB, and per-token log-probs materialised T × 128,000 logits plus a log-softmax copy.
- Fix: reads run with `logits_to_keep=1`, capture the final-normed hidden states, and compute per-token log-probs and top-k in chunks of 256 positions through the tied output layer. Verified against full logits: max log-prob difference 4e-6, identical top-k. Two full traces of 3,294 and 3,564 tokens with all positions: 84 s.
- `fwd wait` now says when the server process is down.
- New `scripts/watchdog.py` (running detached, PID in `queue/watchdog.pid`): restarts the server within about 30 s if its process dies. It requeues the interrupted request. A request that was running during two crashes is moved to `failed/`, not charged, with an explanation, so a crashing request cannot loop.
- The server was down for about 15 minutes before this was noticed. Pass 2 was requeued (not charged).

### 2026-11-05 — gen 4: disk-full incident
- The per-session disk allowance ran out while llama-quantize built a Q8_0 GGUF: claude pass 6 and ed_a pass 5 failed (not charged).
- Freed about 9 GB by deleting the regenerable scratch tensors of finished generations 1–3 (`work/*.safetensors`, large `work/*.npy`, editors' staging `edited/` dirs). Candidates' frozen tensors are kept.
- `wsp/gen.py build_gguf` now evicts old cache entries *before* building (one build needs about 5.3 GB of scratch) and removes its scratch files in a `finally`, so a failed build no longer leaves 3–4 GB behind. The running server picks this up at its next restart.
- claude pass 6 was requeued unchanged. ed_a had already deleted its pass-5 snapshot and resubmits the pass itself.

### 2026-11-06 — gen 4: container restart
- The session container restarted; files survived, but processes (eval server, llama-server, watchdog, the ed_a sub-agent) did not.
- Restarted with `start_server.sh` (which requeued ed_a's running final pass gen_004_ed_a_p06_0ebbf0, uncharged) and the detached watchdog. ed_a was resumed from its transcript.
- claude's work was complete before the restart (candidate gate_wrapup_L20 submitted, report written).

### 2026-11-06 — gen 4: editors finished
- **claude:** candidate `gate_wrapup_L20` (L20 n5414 gated wrap-up neuron, 6,139 weights). DEV +9.4 (SE 6.9). HOLDOUT A +3.1 (SE 3.1). HOLDOUT B +9.4 (SE 5.2). The 64 holdout questions combined: +6.3 pp (SE about 3.0), +4/−0. Report: `runs/gen_004/editors/claude/report.md`.
- **ed_a:** found a sparse commit circuit: L18 n5407 writes Therefore/Hence/Thus, and L19 n50 and L22 n3686 suppress it. Its edit E3 (boost n5407's up-projection along the conclusion direction and zero both suppressors) changed behaviour (commit rate after a conclusion 0.23 → 0.29–0.31, fewer re-checks) but not accuracy. DEV +12.5 (SE 7.4); HOLDOUT +6.25 (SE 4.3); HOLDOUT2 −6.25 (SE 6.2); combined holdouts 0.0 (SE 3.9). No candidate was submitted (D37). Report: `runs/gen_004/editors/ed_a/report.md`.
- Selection queued: champion (base) and `claude/gate_wrapup_L20` on the 96 hidden selection questions (thinking protocol D30).
- Model findings for both editors, with a circuit diagram, were added to `docs/MODEL_FINDINGS.md`.

### 2026-11-06 — gen 4: selection and decision
- Parent (base) on the 96 hidden selection questions, thinking protocol: **42.71%** (SE 5.05); 51% cut at the cap; 87.2% correct when finished; mean 2,385 tokens.
- `claude/gate_wrapup_L20`: **45.83%**. Paired difference **+3.12 pp (SE 3.8, z 0.82)**, 8 gained and 5 lost. Cut share 51.0% → 43.8%; mean tokens 2,385 → 2,236.
- **Decision: no candidate passed (z < 2). The base model stays champion (elitism).**
- Breakdown by trace status (champion → candidate):

  | Champion | Candidate | Count | Correctness |
  |---|---|---|---|
  | cut | finished | 13 | 8 became correct |
  | finished | cut | 6 | 5 had been correct |
  | finished | finished | 41 | every answer kept its correctness; mean tokens 1,564 → 1,381 |
  | cut | cut | 36 | — |

- The losses are not caused by the edit forcing re-checks. Two of the six finished → cut traces never state an answer at all, so the gate cannot have fired there.
- **Paired traces diverge early.** All 96 pairs differ, and the first differing character has a median position of 344 (quartiles 189 and 649); only 1 of 96 diverges right after an answer statement. So pairing barely reduces variance in thinking mode: after the first differing token, the two traces are close to independent samples.
- Ruled out as a cause: quantisation. Our `llama-quantize` of the official BF16 GGUF is byte-identical to the official Q8_0 in all 287 tensors.
- **Confirmed cause: llama.cpp batched generation is nondeterministic.** The unedited base model was rerun with the same Q8_0 file and the same seeds on 16 trace-bank questions, with the cache off and 16 slots (`runs/gen_004/determinism_base16.json`).
  - **0 of 16** traces were reproduced. The first difference fell at characters [18, 18, 27, 37, 76, 81, 90, 94, 249, 264, 304, 313, 447, 448, 470, 1032], the same range as the candidate-versus-champion divergences. One of the 16 answers flipped correctness.
  - The likely mechanism is that the rounding of the CPU kernels depends on how many sequences share a batch.
  - Consequences:
    - In thinking mode, the gen-4 'paired' comparisons were in effect comparisons of independent samples, and the per-question variance is close to that of independent draws. The z statistic is still valid, because the SE was computed from the observed paired differences, but it has little power.
    - The edit did not cause the early divergence.
- Being tested: whether single-slot generation (`-np 1`) is deterministic, and its speed. If it is, unchanged traces stay identical and pairing removes most of the noise.
- Found and fixed while investigating: the doubled `<think>` in read-pass contexts (D39).
- **Single slot (`-np 1`) is deterministic:** 4 of 4 base traces reproduced byte for byte, at 10.9 tok/s (`runs/gen_004/determinism_np1.json`). Unusable at that speed.
- **Batched determinism test** (16 slots, 16 trace-bank questions run twice with the second submission order shuffled, cap 600 tokens):

  | build | identical | tok/s |
  |---|---|---|
  | default build (llamafile sgemm + CPU repack on) | 6 / 16 | 39.4 |
  | `build-det` (`-DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF`) | 7 / 16 | **91.2** |

  - Turning off sgemm and repack does not make generation deterministic.
  - It does make generation **2.3× faster** on this host (no AMX). The repacked and sgemm Q8_0 paths are slow here.
  - Being tested next: separate KV per slot (`-no-kvu`) with prompt-cache reuse off (`cache_prompt=false`), with and without flash attention.
- `wsp/gen.py` gained `EXTRA_ARGS` (extra llama-server arguments) and `CACHE_PROMPT`. Both default to the previous behaviour.
- **Resolved: a deterministic and faster engine (D40).**
  - The `build-det` build with `-no-kvu -fa off` and `cache_prompt=false` reproduced 16 of 16 traces at cap 600 (78 tok/s).
  - It also reproduced **16 of 16 full-length (3,072-token) traces with different batch neighbours** (`determinism_mixed_full.json`; accuracy 7 of 16 both times), at 38–46 tok/s.
  - It is now the default engine (`ENGINE_TAG det1`; the server was restarted).
  - The trace bank is being regenerated under it.
- **What this means for gen 4:**
  - The selection test of `gate_wrapup_L20` (+3.1 pp, z 0.82) is valid but had little power, because about 30 percentage points of the per-question noise came from sampling divergence unrelated to the edit.
  - Under `det1` the edited and unedited traces stay identical up to the first token where the edit actually changes the sampled choice. So only the questions the edit really touches contribute variance.

## 2026-11-06 — Trace bank under `det1`; generation 5 started
- Bank regenerated under `det1`: same 96 ids, 3,468 s (old engine: 8,190 s). Accuracy **38.5%** (SE 5.0); cut 54.2%; 84.1% correct when finished; mean 2,372 tokens (1,544 when finished). This matches the old bank (37.5%, cut 55%, 83.7%).
  - Truncated traces with an in-think answer statement: **9 of 52** (4 correct), against 17 of 53 in the old bank with the same regex. This is sampling variation between the two engines, so the population the gen-4 gate targets is small and variable.
  - Cut share by subject: law 10/12, health 8/11, physics 7/13, engineering 4/5, economics 5/10. No truncations in biology, computer science or history (n = 3, 3, 2).
- Generation 5 created (D41). `carry/gate_wrapup_L20` and the champion are queued for selection on the 128 hidden questions at priority 9.

## 2026-11-06 — Generation 5, editor `claude`, passes 1–3; carried candidate's selection result
- **Pass 1 (read, forced-answer probe on 52 cut traces).**
  - Never-concluding traces give a correct argmax 19–26% of the time (chance 10%), with a mean confidence of 0.8. Only confidence ≥ 0.95 is informative (41% correct).
  - The belief drifts between 30% and 65% of the trace. These traces have not converged; they are not hiding a correct answer.
- **Pass 2 (read, residual at all positions).**
  - There is no clean length signal in the residual. The "late" direction reads wrapping-up content and saturates by about 1,500 tokens.
- **Pass 3 (read, attention from the last token).**
  - Sink-dilution heads (L18 H0/H3/H5/H6/H7, L2.H2) track absolute length (summed d' 2.6), but the head outputs and the residual barely carry it (d' ≤ 1.9 and ≈ 0.4).
  - Written up in MODEL_FINDINGS.
- **Scope decision.** A length-triggered wrap-up is dropped as out of scope ("ends the thinking regardless of the problem"), even though it would likely gain.
- **Selection of the carried gen-4 candidate `gate_wrapup_L20`** on 128 hidden questions, paired, `det1`:
  - Parent 48 correct, edited 60: **+9.4 pp, z 3.6, +12/−0**. Cut 70 → 56.
  - Under the old nondeterministic engine the same edit had measured +3.1 (z 0.82). Removing the sampling noise turned an inconclusive result into a clear one.
  - No trace is byte-identical, because the repurposed neuron is slightly active everywhere. Even so, every flipped answer is a win.
- **Next (`claude`, passes 4–6).** Dose-response of this commit neuron on a fixed DEV32 drawn from the bank: carry exact versus a stronger write (target −15).

## 2026-11-06 — Generation 5, editor `claude`, passes 4–6 and submission
- **Dose-response of the commit neuron on bank questions** (paired; the parent side reproduces the bank byte for byte):

  | set | edit | result |
  |---|---|---|
  | DEV32 | carry, target −9 | +3.1 pp (SE 5.5), +2/−1 |
  | DEV32 | target −15 | +3.1 pp (SE 7.1), +3/−2 |
  | DEV32 | −15 against carry | +3/−3 |
  | HOLDOUT32 | target −15 | −3.1 pp (SE 3.1), +0/−1 |

- **No edited trace stays identical.** The first difference comes at a median of 300–550 characters, so for this edit pairing gives little noise reduction.
- **Submitted** `claude/gate_wrapup_L20_d15` as the dose-response point. Selection measures it alongside the carry (−9).

## 2026-11-06 — Generation 5, editor `ed_a`: finished (summary of its hand-back)
- **Target: hypothesis churn.** After an uncertainty statement ("Not sure.", "Hmm."), the parent opens another speculative paragraph ("Alternatively, maybe ...", "Perhaps ...", "Another approach ...") instead of committing.
  - P(churn opener) at breaks is 0.051 in cut traces and 0.008 in finished ones; the gap is already present in the first quarter of the trace.
  - Ruled out as causes: re-copying options and repeated option sweeps (same rate per token in both kinds of trace).
- **Candidate `ed_a_g5_churn_hence_L23`.** It repurposes L23 n74 (the quietest L23 neuron; 6,143 weights).
  - Gate: a churn-break direction (cross-validated AUC 0.91), threshold at the 55th percentile of churn-break scores.
  - Output: Hence/Therefore minus the churn openers, clipped to the Q8_0 block absmax.
  - Read pass: at firing churn breaks, P(Hence/Therefore) 0 → 0.39 and P(churn) 0.28 → 0.055. Elsewhere the mean |Δlogp| is 0.0004.
- **Paired generate results:**

  | set | result |
  |---|---|
  | DEV32 | +9.4 (SE 5.2, +3/−0) |
  | HOLDOUT bank 32 | 0.0 (SE 4.5, +1/−1) |
  | HOLDOUT fresh 32 | +9.4 (SE 5.2, +3/−0) |
  | holdouts combined (64) | **+4.7 (SE ≈ 3.4, +4/−1)** |

- **Engineering finding: the Q8_0 block artefact.** A new w2 column larger than its Q8_0 block neighbours raises the block scale. llama.cpp then quantises the other 31 neurons in the block more coarsely, and every trace diverges (`ed_a`'s pass 3).
  - Fixed by clipping to 0.98 × the block absmax.
  - Even then, removing any neuron's original tiny contribution flips sampled tokens early, so neuron-repurposing edits keep resampling noise under `det1`. This matches `claude`'s observation for n5414.
- `ed_a` reported that it could not write its own `report.md` (subagents may not write report files). Its full report is in its hand-back message and in `notes.md`.
- Selection queued for `claude/gate_wrapup_L20_d15` and `ed_a/ed_a_g5_churn_hence_L23` (the champion and carry results already exist).

## 2026-11-06 — Generation 5 decision: the first accepted edit
| candidate | acc (128 hidden) | diff | SE | z | +/− | cut share | tokens |
|---|---|---|---|---|---|---|---|
| champion (base) | 37.50 | | | | | 54.7% | 2,265 |
| **carry/gate_wrapup_L20** (gen-4 `claude`, target −9) | **46.88** | **+9.38** | 2.6 | **3.61** | **12/0** | −10.9 pp | −152 |
| claude/gate_wrapup_L20_d15 (target −15) | 43.75 | +6.25 | 3.1 | 2.02 | 12/4 | | |
| ed_a/ed_a_g5_churn_hence_L23 | 40.62 | +3.12 | 1.9 | 1.64 | 5/1 | | |

- **Winner: `carry/gate_wrapup_L20`.** It is now the champion `gen005_carry_gate_wrapup_L20` (D42), the first edit the loop has accepted in five generations.
- **Dose-response.** The stronger write (−15) gains less (+6.25) and loses 4 answers that −9 keeps. The original gen-4 tuning was close to the useful maximum.
- **Queued (D43).** Held-out confirmation in thinking mode (MMLU-Pro 128 and GPQA 198, for base and champion), then the champion's trace bank for generation 6.

## 2026-11-06 — Held-out confirmation runs (D43): first results and an out-of-memory incident
- **Base, held-out MMLU-Pro, 128 questions, thinking mode, `det1`:** accuracy **33.59%** (SE 4.2); cut 56.2%; 76.8% correct when finished; 2,427 tokens on average.
- **Incident.** The base GPQA run failed with `RemoteDisconnected`.
  - The kernel log shows the memory cgroup's OOM killer stopped `llama-server` at 10.4 GB resident (15 GB limit, shared with the PyTorch server and the session).
  - The job was requeued unchanged. Already generated answers are kept in the generation cache.
  - If it recurs, both GPQA runs will use a smaller batch under a separate engine tag, so they remain paired with each other.

## 2026-11-06 — Held-out MMLU-Pro confirms the champion (D43)
Thinking mode, `det1`, 128 untouched held-out questions (`heldout-fixed` order), paired with the base run above:

| model | acc | SE | cut share | acc if finished | mean tokens | tokens if finished |
|---|---|---|---|---|---|---|
| base | 33.59 | 4.2 | 56.2% | 76.8% | 2,427 | |
| **gen005_carry_gate_wrapup_L20** | **40.62** | 4.3 | 52.3% | 85.2% | 2,313 | 1,482 |

- **Paired: +7.03 pp (SE 2.5, z 2.8), 10 wrong → right, 1 right → wrong.** Cut share −3.9 pp (SE 3.2); −113 tokens (SE 46).
- The gain on fresh questions is close to the selection estimate (+9.4), so the selection win was not luck of the draw.
- **Where the gain comes from.** 9 of the 10 gained questions were cut at the cap in the base run and finish correctly under the champion (1,750–2,930 tokens). The 10th changed its answer in a finished trace (C → E). The one loss is the reverse: finished and correct in base, cut under the champion.
  - The net cut share falls only 3.9 pp because other traces moved in both directions without changing the grade (all wrong either way).
  - So the edit rescues correct-but-unfinished reasoning, as in selection; the higher "accuracy if finished" (76.8% → 85.2%) is a consequence, not a separate effect.
- The champion GPQA run crashed with `RemoteDisconnected` (the same OOM kill as the base run); see the next entry.

## 2026-11-06 — Out-of-memory cause found and fixed (D44)
- The champion's GPQA run was also killed (`RemoteDisconnected`), at 62 of 198 answers; the base run had reached 50.
- **Cause:** llama.cpp's host-memory prompt cache (`--cache-ram`, 8 GB by default). It saves the state of every finished slot for possible reuse, although our requests never reuse prompts (`cache_prompt=false`). GPQA's long prompts and full-length traces fill it fastest.
- **Fix:** `-cram 0`. Determinism check on 32 trace-bank questions with 16 slots: **32 of 32 byte-identical** to the `det1` bank (1,474 s). Results computed before and after the fix can therefore be mixed and paired; the engine tag stays `det1`.
- The cached answers are kept and the two GPQA jobs are requeued (champion first, then base, then the champion trace bank).

## 2026-11-06 ~22:30 — Container restart onto a different host
- The container restarted at about 22:30 and stopped every process. At that point the champion's GPQA run had 111 of 198 answers cached; base GPQA has 50.
- **The new host has AMX and AVX-512 (bf16, VNNI) again**; the previous one (since 2026-11-05) had neither (D36). The `det1` llama.cpp binary was compiled with `GGML_NATIVE` on the old host, so it uses the same instructions here.
- **Check:** 32 of 32 trace-bank traces byte-identical to the `det1` bank under the new host (1,599 s; `runs/gen_005/engine_checks/cram0_determinism_newhost_amx.txt`). Generation results from both hosts can therefore be mixed and paired.
- The PyTorch side (score and read passes, float32 since D36) may now use different matmul kernels, so recorded activations can differ in the last bits from those computed on the previous host. That matters only for comparing activations across hosts, not for generation results.
- Queue, watchdog and memory log restarted; the interrupted champion GPQA job resumes from its cache.

## 2026-11-07 — The champion's trace bank, and generation 6 (D46)
- **GPQA, champion:** all 198 answers generated (thinking mode, `det1`); the paired comparison with base follows when the base run (priority 6) finishes.
- **Champion trace bank** (same 96 practice questions, `det1`): accuracy 38.5% (the same as base), cut 53 of 96 (base 52), 37 of 43 finished correct (86%), 1,470 tokens per finished answer (base 1,544).
  - **Paired with the base bank: +5/−5**, 7 cut → finished and 8 finished → cut; no trace byte-identical (the repurposed neuron diverges every trace early).
  - So on these 96 questions the champion shows no gain, against +9.4 (selection, +12/−0) and +7.0 (held out, +10/−1) on fresh questions. Pooled over the 352 questions measured under `det1`: **+27/−6**.
  - The 4 regressions where base finished correctly and the champion is cut (8262, 7373, 5325, 8198): in three, the champion states the right answer mid-trace (43–66%) and keeps deliberating.
  - 18 of the champion's 53 cut traces state an answer inside the thinking (10 correct); base had 7. About half state it in the last 10% (too late); the rest state it at 19–66%, mostly hedged ("Thus likely answer is (E) …") or firm and then re-verified ("The answer is (G). But we must double-check …").
- **Generation 6 created (D46):** parent `gen005_carry_gate_wrapup_L20`.
  - Editors `claude` (never-concluding knowledge questions: law and health) and a fresh `ed_a` (answer statements that do not close).
  - Carried candidate `carry/churn_hence_L23` (gen-5 `ed_a` edit, stacked).
  - Parent and carry selection queued at priority 9.
  - `ga select` gained `--priority`. `ga brief` falls back to an editor's `notes.md` when `report.md` is missing.

## 2026-11-07 ~03:00 — Disk-full incident (D47)
- Both editors' 4th passes (generate, gen 6) failed at `llama-quantize` with 230 MB of disk left; failed passes are not charged and both were resubmitted.
- Freed 7.6 GB: two old gen-5 Q8_0 builds and 0.9 GB of activation arrays from finished generations; the GGUF cache now keeps at most 3 edited builds.
- The base GPQA job (priority 6) had started in the gap; the server was restarted so that editor passes run first again.

## 2026-11-07 ~05:20 — Second container restart
- Same host type (AMX). All processes stopped, including the `ed_a` sub-agent and `claude`'s pass 6 (generate, running).
- Queue, watchdog and memory log restarted; the interrupted pass was requeued (not charged twice). `ed_a` was resumed from its saved transcript with a note about the restart.

## 2026-11-07 — Generation 6 editors finished
- **`claude` → `finalist_commit_L21`** (one repurposed neuron, L21 n6823). Target: cut traces that narrow to two finalists ("either F or G") and then oscillate until the cap (18 of the parent's 53 cut bank traces; finished traces rarely narrow, 4 of 43).
  - Forced answer at the first two-finalist paragraph end: 8 of 18 correct (7 of 11 when gold is one of the two; probability on the pair 0.99). Forced at the cap: 10 of 18.
  - Gate: mean-difference direction at the L21 MLP input between breaks ending two-finalist paragraphs and other breaks (leave-one-trace-out AUC 0.86). Output: commit openers minus deliberation openers.
  - Read: where the gate fires, deliberation openers collapse ("Option" 0.80 → 0.04); elsewhere mean |Δlogp| 0.004 per token.
  - Generate (paired): DEV32 +9.4 (+3/−0); HOLDOUT32 +6.25 (+2/−0); HOLDOUT32-B 0.0 (+1/−1). Holdouts combined (64 fresh): **+3.1 (SE ≈ 3.1, +3/−1)**. All 5 DEV/HOLDOUT32 gains are narrowing traces that now finish with the right finalist; the HOLDOUT32-B changes come from early divergence.
- **`ed_a` → `ed_a_g6_short_answer_L23`** (three repurposed L23 neurons: n4526, n7602, n810). Target: after `</think>` the parent re-derives the solution in the visible answer (~250 tokens) before "The answer is (X)". In 42 of 43 finished bank traces it is the answer already settled in the thinking. 5 of 53 cut bank traces were cut while writing that answer, 4 of them with the right answer already settled.
  - The edit makes the visible answer exactly `"The answer is (X)"` followed by the end of message, with the letter copied from the thinking.
  - Generate: DEV32, an earlier two-neuron version: −9.4 (+3/−6; the answer opener sometimes led to a quoted explanation without an answer). Fresh 32, the final three-neuron version: **+6.25 (SE 4.3, +2/−0)**, cut 53% → 41%, −265 tokens; all 19 finished answers have exactly that form.
  - Scope judgment: admitted (D50).
  - Its report also found a **grading inconsistency**: a trace cut after writing "answer is (X)" would have counted as correct. No existing row was affected; fixed (D49).
- Selection queued at priority 9: parent, `carry/churn_hence_L23`, `claude/finalist_commit_L21`, `ed_a/ed_a_g6_short_answer_L23`. Base GPQA after them (D48).

### 2026-11-07 08:35 UTC: gen 6 selection, parent done
- Parent (gen005_carry_gate_wrapup_L20) on the 128 selection questions (think, det1): 69/128 = 53.9%, cut 48/128. No cut-and-correct rows, so the D49 grading fix does not change this number.
- carry/churn_hence_L23 running; claude/finalist_commit_L21 and ed_a/ed_a_g6_short_answer_L23 pending, then base GPQA (priority 10).
- The wait task hit its 2 h background limit and was re-armed; nothing failed.

### 2026-11-07 10:40 UTC: gen 6 selection runs invalid, fixed and requeued (D52)
- The carry and `claude` selection outputs were byte-identical to the parent's (69/128, cut 48, every trace identical).
- Cause: the worker's graceful restart (`os.execv`, for the D49 grading fix) left its llama-server child, serving the parent's GGUF, on the port. New servers for the edited models failed to bind; the old one answered.
- `ed_a`'s run was in progress on the same orphan; stopped by killing it.
- Guards added (D52); invalid outputs quarantined; the three runs requeued. The base GPQA run (priority 10) had become next and started on a verified fresh server; selection follows it.
- Audit of all generation caches: no other wrong-model result.
- Lesson: a health check proves that *a* server answers, not that it is *ours* with the right weights.

### 2026-11-07 12:05 UTC: GPQA transfer of the gen-5 parent (held out, thinking, det1)
| model | correct / 198 | cut | mean tokens | acc. when finished |
|---|---|---|---|---|
| base | 25 (12.6%) | 168 | 2,900 | 0.83 (25/30) |
| gen005_carry_gate_wrapup_L20 | 33 (16.7%) | 158 | 2,854 | 0.83 (33/40) |
- Paired: **+4.04 pp (SE 2.13, z 1.90), +13/−5**. Same direction as on MMLU-Pro held out (+7.0, z 2.8), but short of z 2. The gain comes from finishing more traces; accuracy when finished is unchanged.
- No cut-and-correct rows in either file (D49 has no effect).
- Caveat: GPQA traces almost never finish within the 3,072-token cap (85% cut for base), so this mostly measures whether a trace concludes. Both models score below the 25% chance level because cut answers count as wrong.
- Base run served by a verified fresh server (D52 guards active).

### 2026-11-07 14:05 UTC: gen 6 selection re-run, carry done
- carry/churn_hence_L23: 68/128 vs parent 69/128, **−0.78 pp (SE 2.60, z −0.30), +5/−6**, cut 50 vs 48. Only 12 of 128 traces identical to the parent's, which confirms the edited weights were served (D52 guards).
- claude/finalist_commit_L21 running on the verified GGUF; ed_a/ed_a_g6_short_answer_L23 pending.

### 2026-11-07 15:45 UTC: generation 6 decision (D51): the champion survives
| candidate | acc | diff vs parent (53.91) | SE | z | +/− | cut (parent 48) | mean tokens (parent 2,109) | think identical to parent | median first divergence |
|---|---|---|---|---|---|---|---|---|---|
| ed_a/ed_a_g6_short_answer_L23 | 54.69 | +0.78 | 3.2 | +0.24 | +9/−8 | 47 | 1,929 | 11/128 | 14% of the trace |
| claude/finalist_commit_L21 | 53.91 | 0.00 | 4.0 | 0.00 | +13/−13 | 52 | 2,091 | 3/128 | 7% |
| carry/churn_hence_L23 | 53.12 | −0.78 | 2.6 | −0.30 | +5/−6 | 50 | 2,085 | 16/128 | 22% |
- All three outputs checked against D52: distinct traces, verified GGUFs, no cut-and-correct rows.
- **Why the editors' fresh-holdout gains (+3 to +6 pp) vanished: resampling noise.**
  - Every edit repurposes a neuron, which removes that neuron's original output. That slightly changes all logits, and with temperature 0.6 a changed sampled token soon follows. `claude`'s traces diverge from the parent's at a median of 7% of the trace (84 of 128 before 10%), far before its gate can fire (two-finalist paragraphs come at ~58%).
  - Its 26 swaps: all 13 losses are parent-finished traces that the edited model now runs to the cap; 11 of 13 gains are parent-cut traces it now finishes. Only 3 of the 12 parent traces that were cut and narrowed to two finalists became correct, and those also diverged early (2-12%), so even they cannot be credited to the gate.
  - With SE 2.6-4.0 on 128 questions, effects of +2 to +4 pp (the size the editors' mechanisms predict) cannot reach z 2. The gen-5 champion passed because its effect was larger (+9.4, z 3.6).
- Generation-6 report: `runs/gen_006/REPORT.md`.

## 2026-11-07 ~16:00: generation 7 created (D53)
- Parent unchanged (`gen005_carry_gate_wrapup_L20`). Editors `claude` and a fresh `ed_a`; 6 passes, ≤ 1 candidate each.
- The new brief section on generation 6 states the power arithmetic: on 128 paired questions, +4/−0 passes (z 2.0), but early divergence produces 10-26 random swaps. Editors are asked for edits that are exactly silent off-target, verified by counting byte-identical traces.
- Parent selection queued at priority 9.

### 2026-11-07 — gen 7, editor `claude`: pass 2 analysis and edit v1; disk housekeeping
- Pass 2 (read, parent, MLP inputs at L22/L23 on 32 bank traces): the two-finalist break direction separates at L22 (LOTO AUC 0.864) and L23 (0.859), as at L21 in gen 6 (0.86).
- New finding: the bias-proxy gate fires on the first prompt tokens of every text. These are low-norm, constant attention-sink tokens: 'user' at L22; BOS, `<|im_start|>` and `\n` at L23. Such a gate perturbs every question through the KV cache. The fix projects the gate row off the 4-dim subspace of those tokens and subtracts a small sink penalty.
- Edit v1, L22 n1786 (the second-lowest removal impact, 0.0013): threshold 7.5, k 6, w2 column norm 1.47.
  - It fires on 79% of two-finalist breaks, 13.7% of other breaks and 0.27% of all tokens.
- The trace bank reproduces the parent byte-identically: gen-6 pass-4 originals matched 32/32. An edited-only generate on bank questions is therefore paired for free (48 questions per pass instead of 32).
- Pass 3: edited generate on DEV32 plus 16 further bank questions.
- Disk: deleted the per-pass weight snapshots of generations 1–6, 2.7 GB in total. They were gitignored; submitted candidates keep their weights. Hashes are recorded in runs/snapshot_hashes_gen1-6.txt.

### 2026-11-07 — gen 7, editor `claude`: pass 3 (edit v1 generate) and divergence diagnosis
- Edit v1 (L22 n1786 two-finalist gate) on 48 bank questions, compared paired against the bank (identical to the parent):
  - DEV32: −6.2 pp (+1/−3);
  - EXTRA16: 0;
  - byte-identical traces: 0 of 48; median first divergence at 5% of the trace.
- The mechanism still shows: a commit opener follows a two-finalist paragraph in 5/18 cases, against 3/39 for the parent.
- Diagnosis: the new gate is silent before divergence in 31 of 32 traces (0.000 up to the first diverging token). Divergence comes from removing the original neuron's small output (~0.02 per token).
- The brief's goal, exact silence, is impossible under the rules:
  - no neuron is Q8-silent;
  - no w2 column quantises to zero;
  - the only exactly silent carrier found is forbidden. That carrier is the tied embedding row of a never-used token: top-k 20 makes logits outside the top 20 irrelevant, and the token never appears in the context.
- Pass 4 completes the paired test on the remaining 48 bank questions.

### 2026-11-07 — gen 7, editor `claude`: pass 4 and outcome (no candidate)
- Edit v1, paired against the bank on all 96 questions: 37 → 38, +1.0 pp (SE 2.8), +4/−3; cut 53 → 50; byte-identical 1/96. REST48 gives +6.2 pp, but all three of its gains are in non-target traces (resampling).
- The mechanism is confirmed: a commit word follows a two-finalist paragraph in 10/29 cases, against 3/39 for the parent. The accuracy effect is within noise.
- Noise floor measured: correctness flips on 7/96 questions and cut status on 13/96 under resampling, which gives SE ≈ 2.4 pp at 128.
- Decision (editor): no candidate submitted (expected z ≈ 0.3). Passes 5–6 left unused. Report: runs/gen_007/editors/claude/report.md.

### 2026-11-07 — gen 7: ed_a hand-back; selection queued
- ed_a submitted one candidate, `ed_a_g7_short_answer_dupfold_L23`, in layer 23.
  - It frees slots in a cluster of four near-duplicate L23 neurons (w1/w3/w2 cos ≥ .9998, activation corr .9999) by merging their w2 columns into one twin.
  - It hosts three gated units in the freed slots: `</think>` → `"The`, `"The` → ` answer`, and `)".` → `<|im_end|>`. The visible answer becomes `"The answer is (X)"`.
  - Results: DEV32 +15.6 pp (+6/−1; 4 gains from the edit itself), HOLDOUT A 0.0 (+1/−1), HOLDOUT B −6.25 (+1/−3).
  - Byte-identical traces were only 2–6 of 32; median first divergence at 13–19% of the trace.
  - ed_a expects 0 ± 3 pp in selection.
- ed_a independently reached the same conclusion as `claude`: under Q8_0 with sampling, exact silence is unattainable. A per-token change of ~1e-4 already flips an early near-tie token.
- New finding (ed_a): L23 contains near-duplicate neurons ({1228, 3401, 3793, 6986}, {331, 5105, 881}, {5821, 2400}). Merging them frees slots with 4× less output change than removing the quietest neurons.
- Process note: ed_a's attempt to create `report.md` was refused by the tool's permissions. It appended its report to `notes.md` (`## REPORT`) instead; `ga brief` falls back to notes.md. The orchestrator did not write the file on its behalf.
- `ga select --priority 9`: 1 candidate on the 128 hidden questions (the champion run is already done).

### 2026-11-07 — gen 7 selection and decision (D54)
| candidate | acc | diff vs champion (46.09) | SE | z | +/− | cut |
|---|---|---|---|---|---|---|
| ed_a_g7_short_answer_dupfold_L23 | 49.22 | +3.12 | 2.9 | 1.08 | +9/−5 | 58 → 52 |

- D52 validity check: 1/128 traces byte-identical; 74/76 finished answers in the edit's short form.
- Event analysis: the champion closed the thinking and was then cut inside the answer in 5 traces. The edit rescued 1 of them. The other 8 gains and 5 losses are resampling.
- Champion survives (elitism). Cosmetic: `ga report` labels generate passes "read None texts".

### 2026-11-07 — Generation 8 created (D55)
- Parent: `gen005_carry_gate_wrapup_L20`. Editors: `claude` and a fresh `ed_a` (background sub-agent), 6 passes and ≤ 1 candidate each; 128 hidden questions.
- Brief: `prompts/editor_task_think_g8.md`. It sets a size target (≈ +6 pp) and asks editors to stack mechanisms.
- Carry: `stack_shortans_L23_finalist_L22`, the union of generation 7's L23 short answer (ed_a) and L22 two-finalist gate (claude).
- `ga select --priority 9`: champion and carry queued.
- Bank cut rate by category (for the briefs): law 9/12, health 9/11, physics 6/13, math 5/7, economics 5/10, psychology 5/9, engineering 4/5. Cut traces contain fewer numbers than finished ones (4.9 vs 6.5 per 1k characters), so long computation is not the main cause of cuts.

### 2026-11-08 — container restart during gen 8
- The container restarted. Lost: the forward-pass server, the watchdog, the `ed_a` sub-agent and the editors' wait loops. Files and git were intact; the champion's gen-8 selection run (128 rows) had completed.
- Recovery:
  - `./start_server.sh` requeued the interrupted job (`claude` p01) without counting it as a crash.
  - The watchdog was restarted.
  - The `ed_a` agent was resumed from its saved transcript.
- Queue after recovery: `claude` p01, `ed_a` p01, then the carry selection run.

### 2026-11-08 — gen 8: disk-full build failures, passes re-submitted
- `claude` passes 3–4 and `ed_a`'s pass 3 failed in `llama-quantize`; failed passes are not charged. Free disk was 1.3 GB, while a GGUF build needs ~5.3 GB of scratch.
- Freed 1.9 GB of `claude`'s own files:
  - the pass-1 L23 activations (the fitted direction is saved);
  - four 361 MB per-pass weight snapshots of finished or failed passes.
- Re-submitted `claude` passes 3–4; `ed_a` re-submitted its own pass. The carry selection run is now running.
- The per-pass snapshots (361 MB each with two edited layers) are the main recurring disk cost. Snapshots of finished passes could be deleted automatically.

### 2026-11-08 — gen 8: carry selection, and claude pass 3
- Selection, 128 hidden questions. The champion scores 41.41 on this generation's hidden set.

| candidate | acc | diff vs champion | SE | z | +/− | cut |
|---|---|---|---|---|---|---|
| carry `stack_shortans_L23_finalist_L22` | 46.88 | +5.47 | 2.3 | 2.38 | +8/−1 | 47.7 → 40.6 % |

- Validity (D52): 1/128 traces byte-identical, so the edit is active. Mean tokens 2,190 → 2,019.
- The carry passes z ≥ 2. Taken separately, neither part passed in gen 7: the L23 short answer had z 1.08; the L22 two-finalist gate was never submitted. So stacking two weak mechanisms crossed the threshold.
- `claude` pass 3: carry plus the new L23 n881 commit-opener unit, generated on DEV32 + EXTRA16 and paired against the bank.
  - Result: +14.6 pp on 48 (SE 6.7, z 2.19), +9/−2; cuts 27 → 19.
  - DEV32 is in-sample: all 22 fit traces come from it (+18.8). The 16 out-of-sample questions gave +6.2 (+1/−0).
  - Pass 4 runs on REST48, which is fully out of sample.

### 2026-11-08 — gen 8: claude passes 4–6, and a disk guard
- `claude` pass 4 (stack + unit, REST48, fully out of sample): +10.4 pp (SE 4.5, z 2.34), +5/−0. On all 96 bank questions: +12.5 pp (SE 4.0, z 3.13), +14/−2; cuts 53 → 43.
- Passes 5–6 isolate the unit's own effect: the carry alone on REST48 (pass 5) and on DEV32 + EXTRA16 (pass 6), paired against stack + unit.
  - `edited/` held the carry tensors only while each snapshot was taken (6/6 files verified); the unit tensors were restored and verified.
- Disk: 3.3 GB free, and one cache eviction frees only 1.8 GB, so the next build would likely fail again.
  - Deleted `claude`'s untracked gen-7 activation arrays (fwd_01, fwd_02) and two snapshots, ≈ 1.3 GB.
  - Now 4.5 GB free (6.3 GB after eviction).

### 2026-11-08 — gen 8: ed_a hand-back
- Candidate: `ed_a_g8_sweep_end_hence_L23`. Six passes used; report in `runs/gen_008/editors/ed_a/report.md`.
- Mechanism: at the paragraph break that ends a full pass through the options (a "sweep end"), a gated unit writes "Hence" instead of "Thus"/"We"/"Given".
  - The gate is a shrinkage-LDA direction at the L23 MLP input (AUC 0.968), firing at z > 8.
  - The unit sits on L23 n5105, freed by folding n5105 into its twin n331 (cos 0.9995).
- Evidence:
  - All 96 bank questions: +6.2 pp (SE 3.2), +8/−2.
  - 64 fresh holdout questions: −3.1 pp (SE 3.8), +2/−4.
  - ed_a expects 0 to +3 pp in selection.
- Finding: forcing "Hence, the answer is (" at a sweep end is right in 36% of cut traces (they score 0 now) and costs ~1 in 25 finished traces. But only 74 of the 160 inserted "Hence" paragraphs state an answer ("Hence we need …" keeps deliberating).
- Stacking conflict: ed_a's g8 unit and `claude`'s g8 opener unit both use the L23 duplicate group {331, 5105, 881}. `claude` folds n881 into n331 + n5105; ed_a repurposes n5105. They cannot be combined as is; a stack needs a different free slot for one of them.

### 2026-11-08 — gen 8: claude candidate submitted, selection queued
- Unit-vs-carry control (passes 5–6, carry alone, generate). Stack + unit vs carry:

| questions | diff (SE) | z | +/− |
|---|---|---|---|
| all 96 | +11.5 (3.9) | 2.95 | +13/−2 |
| REST48 (out of sample) | +8.3 (4.0) | 2.07 | +4/−0 |

  - 14/96 traces were byte-identical to the carry's.
  - The carry alone vs the parent on the bank: +1.0 (3.5), +6/−5. Its +5.47 on the selection set is probably partly luck.
- Submitted `claude_g8_stack_commit_opener_L23` (carry + L23 n881 commit opener); report in `runs/gen_008/editors/claude/report.md`.
- `ga select --priority 9` queued both candidates (claude, then ed_a) on the 128 hidden questions.

### 2026-11-08 — gen 8 selection: claude candidate
- Result on the 128 hidden questions:

| candidate | acc | diff vs champion (41.41) | SE | z | +/− | cut |
|---|---|---|---|---|---|---|
| `claude_g8_stack_commit_opener_L23` | 46.09 | +4.69 | 3.1 | 1.51 | +11/−5 | 47.7 → 42.2 % |

- It fails z ≥ 2. Paired against the carry on the same 128 questions it is −0.78 (SE 2.8), +6/−7; 16/128 traces are identical to the carry's.
- The bank effect (unit vs carry +8.3 on REST48, +11.5 on 96) did not transfer: the two estimates differ by about 2.5 SE.
- Coverage is smaller in the selection set: 23 of 128 carry traces have a two-finalist statement. Of the 13 swaps against the carry:
  - 5 gains show the unit's commit ("Thus I'd go with A", "Thus answer (A)");
  - the losses include traces that were resampled into the cap (1274, 10813, 11756, 3813), plus 2 commits to a wrong option (646, 10979).
- Lesson: an out-of-sample split of the same 96-question bank is not a full holdout for the decision.
  - The bank is a random sample of practice questions (D33). Correction: an earlier version of this entry said the bank was picked as cut-heavy; it was not.
  - But REST48 had already been looked at in generation 7, and this generation chose its next steps after about 10 looks at bank results (winner's curse).
  - Bank and selection have disagreed before, in the other direction (D46: the gen-5 champion is +5/−5 on its own bank against +9.4 on fresh questions).
  - Treat a bank gain as a screen. Measure on fresh questions before submitting.

### 2026-11-08 — gen 8 decision (D56): the carry stack becomes champion
- `ga decide` result:
  - **WINNER: `carry/stack_shortans_L23_finalist_L22`** (+5.47, z 2.38, +8/−1). New champion: `gen008_carry_stack_shortans_L23_finalist_L22`.
  - `claude`: +4.69, z 1.51.
  - `ed_a`: +3.12 (SE 2.5, z 1.25), +7/−3. Its traces carry 84 "Hence" paragraphs against the champion's 38, so the edit was active; 11/128 traces are byte-identical.
- `ga report` wrote `runs/gen_008/REPORT.md`.
- Queued held-out confirmation at priority 1: MMLU-Pro 128 (thinking), then GPQA Diamond 198 (thinking). Logs: `runs/heldout/think_gen008_{mmlupro,gpqa}.log`.
- The gen-5 champion, for comparison: held-out 40.62% (+7.03 vs base, z 2.8).
- The new champion's trace bank (same 96 practice ids, `det1`) is queued at priority 2 → `runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl`, for generation 9.
- The champion's weights touch L20 (gen 5), L22 and L23 (gen 8).
- MODEL_FINDINGS has a generation-8 section: commit points in long traces, and the champion's three-unit pipeline (diagram).

### 2026-11-08 — held-out confirmation of the gen-8 champion (MMLU-Pro, thinking)
128 untouched held-out questions (`heldout-fixed`), protocol D30, `det1`:

| model | acc | cut | acc if finished | mean tokens |
|---|---|---|---|---|
| base | 33.59 | 56.3 % | — | — |
| gen-5 champion | 40.62 | 52.3 % | 85.2 | 2,313 |
| **gen-8 champion** | **45.31** | **46.1 %** | 84.1 | 2,115 |

- Against base: **+11.72 pp (SE 3.1, z 3.8), +16/−1**; cut share −10.2 pp; −312 tokens.
- Against the gen-5 champion: +4.69 pp (SE 3.5, z 1.34), +13/−7; cut share −6.3 pp; −198 tokens.
- This matches the selection estimate (+5.47), so the gain holds up on untouched questions. The +1.0 on the bank was the low draw.
- Next in the queue: the champion trace bank (running), then GPQA Diamond 198.
- D57: a holdout bank of 96 untouched practice questions is queued at priority 2, behind the main bank. The champion's GPQA run was moved to priority 6 (transfer reporting only), so generation 9 can start sooner.

### 2026-11-08 — generation 9 created (D58)
- New champion bank (same 96 ids): 39.6% (SE 5.0); 47/96 cut; 78% correct when finished; visible answer ~19 characters (gen-5 champion: ~985).
  - Against the gen-5 champion on the bank: +1.0 (+6/−5); cut 10 → finished, 4 finished → cut.
  - These traces are byte-identical to `claude`'s gen-8 pass-5/6 carry-alone generations, as expected under `det1`.
- 13 of the 47 cut traces state an answer inside the thinking (7 correct).
- Brief `editor_task_think_g9.md`; focus in `runs/gen_009/focus.json`; the parent's selection run is queued.
- The holdout bank (D57) is still generating, at priority 2, ahead of editor passes.
- Holdout bank (D57) done: 96 rows, `runs/shared/holdout_bank_gen008champ_cap3072_det1.jsonl`. Its aggregate numbers are not recorded here and not shown to editors, so the editors' final measurement on it stays untouched.
- The champion's GPQA run (priority 6) started before any gen-9 editor pass had been submitted, so editor passes wait ~2.5 h behind it. It was not interrupted, since mid-job restarts caused D52.

### 2026-11-08 17:30 — Gen-8 champion on GPQA Diamond (transfer, thinking mode, 198 questions)
- Champion `gen008_carry_stack_shortans_L23_finalist_L22`: 19.19% (SE 2.8). Cut share 75.8%; accuracy on finished traces 79%; mean 2,774 tokens.
- Paired against base: 12.63 → 19.19, **+6.57 pp (SE 2.0, z 3.23), +15/−2**. Cut answers 168 → 150.
- Paired against the gen-5 champion: 16.67 → 19.19, +2.53 pp (SE 2.3, z 1.09), +13/−8. Cut answers 158 → 150.
- Reading: the gain over base is now significant on GPQA, which was never used for tuning. The step from gen 5 to gen 8 points the same way as on MMLU-Pro but is not significant at n = 198. GPQA still mostly measures whether a trace concludes (76% are cut), so it scores below the 25% chance level.
- Log: `runs/heldout/think_gen008_gpqa.log`.

### 2026-11-08 18:05 — Server OOM-killed by an editor analysis (claude, gen 9)
- My pass-2 analysis (`code/008`) loaded three 69k×2048 float32 arrays plus the 128k-token embedding (~4 GB) while the server ran ed_a's read pass (~10 GB). The 15 GB memory cgroup killed the server (pid 213).
- The watchdog restarted it (pid 3780) and requeued ed_a's job (`gen_009_ed_a_p02`). No pass was charged twice and none was lost.
- Fix: editor analyses use memory-mapped arrays, chunked products and a bf16 embedding (~1 GB peak). This is noted for future briefs as a pitfall: "keep analyses under ~3 GB; the server needs ~10 GB during read passes".
- 18:20 Disk cleanup so the next GGUF builds fit (they need ~5.3 GB of scratch; eviction frees 1.8 GB). Deleted git-ignored scratch only:
  - gen-8 editors' per-pass weight snapshots (ed_a fwd_02–06);
  - the gen-7 ed_a fwd_04 snapshot;
  - my gen-8 `work/unit_edited` (identical to the submitted candidate in `candidates/`);
  - gen-2 ed_b `p1_feats.npz`;
  - my gen-9 pass-1/2 activation arrays.
  - Free space went from 2.4 to 4.1 GB.
- 20:48 `claude` pass 6 (holdout H[48:], gen_009_claude_p06_2ec58a) FAILED in `llama-quantize`. Cause: disk exhausted during the GGUF build; it was not charged.
  - Freed 1.6 GB by deleting old gens' editor `edited/` dirs (gen 4–8) that are byte-identical to a saved candidate. Checked with `diff -rq`; dirs without an identical candidate were kept.
  - Also deleted the pass-5 snapshot. Free space went from 4.4 to 6.0 GB.
  - Requeued as gen_009_claude_p06_8ed421.
  - Recurring cause: with MAX_GGUF_CACHE = 3, the cache holds about 3.7 GB after eviction, and builds need about 5.3 GB of scratch on top. Recommendation: lower MAX_GGUF_CACHE to 2 (only 1 q8 file left after eviction) and delete snapshots of finished passes automatically.
- `claude` holdout pass 5 (H[:48]): Unit A +0.0 pp (SE 5.2; +3/−3; cut 16 → 15). The in-bank +8.3 does not carry over to this half.
- `claude` holdout pass 6 (H[48:]) +4.2 (SE 5.9). Full holdout 96: Unit A **+2.1 pp (SE 3.9, z 0.53, +8/−6)**, cut 42 → 38.
  - Mechanism: 11 of 42 cut traces close right after a conclusion; 7 of those 11 are right.
  - Losses: 5 of 6 are finished traces resampled into new cuts.
  - Submitted as `claude_g9_firm_conclusion_close_L23` (code/014); report in runs/gen_009/editors/claude/report.md.

### 2026-11-08 — gen 9 candidates and selection
- `ed_a` (background agent) submitted `ed_a_g9_sweep_end_hence_answer_L23`, using all 6 passes (2 read, 4 generate).
  - Edit: L23 n5105/n881 freed by folding into duplicate n331. Unit A writes "Hence" at sweep-end breaks; Unit B writes " answer" after Hence/Thus at a sweep end. 14,330 weights.
  - Holdout bank 96: **−2.1 pp (SE 3.9, +6/−8)**. Main bank: +4.2 (SE 3.6). Report: runs/gen_009/editors/ed_a/report.md.
  - Reasons it reports: after "Hence answer" the model hedges ("likely either F or J"); when it does commit, the parent would mostly have reached the same letter anyway.
  - Pitfall: its resubmitted failed pass got the same pass number as a queued one, and that pass's output overwrote its `work/fwd_06` jsonl. The per-question rows of its holdout half B are lost; the summary was recorded first.
- `claude` submitted `claude_g9_firm_conclusion_close_L23`: holdout **+2.1 (SE 3.9)**.
- Neither candidate is significant on the holdout bank. Both still go to selection, per protocol.
- `ga select --priority 9` queued: champion (cached, D35), then claude, then ed_a, on the 128 hidden questions (seed `selection-gen-9`).
- Infra fix (`wsp/fwd.py`): a pass number is now the smallest one not held by a finished or pending pass. It used to be `used + pending + 1`.
  - The old rule: after a failed pass, a resubmission could reuse the number of a pass still in the queue, which then overwrote that pass's `work/fwd_NN` output. This is what happened to `ed_a`.
  - Tested on 4 cases, including ed_a's state (passes 1–4 plus pending p06 → 5). Results are unaffected; it only changes output folder names.

### 2026-11-09 — gen 9 selection and decision (D59)
| candidate | acc | diff vs champion (49.22) | SE | z | +/− | cut (of 128) | identical traces |
|---|---|---|---|---|---|---|---|
| `claude_g9_firm_conclusion_close_L23` | 48.44 | −0.78 | 2.4 | −0.32 | +4/−5 | 51 → 51 | 2/128 |
| `ed_a_g9_sweep_end_hence_answer_L23` | 46.88 | −2.34 | 2.1 | −1.11 | +2/−5 | 51 → 53 | 12/128 |

- D52 check passed: each run has its own weights key, and its traces differ from the champion's.
- Outcome: no candidate passed; the champion survives (elitism). `ga report` wrote `runs/gen_009/REPORT.md`.
- Swap breakdown:
  - claude: 6 cut traces closed (4 right); 6 finished traces newly cut (5 were right).
  - ed_a: 5 closed (2 right); 7 newly cut (5 were right).
- The holdout bank (+2.1 / −2.1) matched selection to within 1 SE; the main bank (+8.3 / +4.2) did not.
- The `ga select` duplicate champion job re-used the cache (0.2 s); there is one champion output.
- MODEL_FINDINGS: new generation-9 section (budget map of cut traces, the statement-end close unit with a diagram, and why close/commit units cancel against resampling).

### 2026-11-09 — cap-sensitivity check queued (D60)
- The user asked where the 3,072-token cap comes from. It is ours (D30), chosen for CPU cost. The paper's Exp 8 used 32,768 (thinking) and 8,192 (no thinking); see `exp8_math/code/evaluate.py` in its logs.
- The container had restarted (uptime 9 min), so the server and watchdog were down. Both restarted at 09:08 (server pid 455) with the new per-job cap code.
- Queued at priority 1: `ga heldout base|champion --n 128 --protocol think --cap 8192`. llama-server runs with `-c 147456` (16 × 9,216).
- Champion weight footprint vs base: 32,756 weights in 9 MLP tensors (L20, L22, L23), about 0.002% of the model.

### 2026-11-09 11:20 — cap-8192 check (D60), interim
- The waiter on both runs hit its 2 h background limit. Nothing failed: base is at 79/128 answers after about 2 h 05 m (the long-thinking questions come last). The champion's run is queued behind it.
- **Determinism check passed early.** All 35 base traces that finished under 3,072 tokens in the 3,072 run and are already in the 8,192 cache are byte-identical. A larger `-c` (147,456 vs 65,536) does not change det1 sampling. So any 3,072 → 8,192 difference comes only from traces that were cut at 3,072.
- Added `scripts/cap_compare.py`: per-run summaries, paired champion-vs-base at each cap, paired 8,192-vs-3,072 per model, the prefix check, and how many cut@3,072 traces finish by 8,192 and how many of those are right.

### 2026-11-09 12:40 — cap-8192 check (D60): base done
Base finished at 12:39 (about 3 h 30 m). Base, held-out MMLU-Pro 128, det1, same seeds:

| cap | acc | cut | acc when finished | median tokens | mean tokens |
|---|---|---|---|---|---|
| 3,072 | 33.59 | 56.25 | 76.79 | 3,072 | 2,426 |
| **8,192** | **56.25** | 21.09 | 71.29 | 3,757 | 4,338 |

- Paired 8,192 vs 3,072: **+22.66 pp (SE 3.71, z 6.1), +29/−0.** It can only gain: all 56 traces finished under 3,072 are byte-identical, and a cut trace simply continues.
- Of the 45 traces cut at 3,072 that finish by 8,192, 29 are right (64%). 27 traces are still cut at 8,192.
- **Implication:** base at 8,192 (56.3) beats the champion at 3,072 (45.3). The champion's held-out gain is a gain *at a 3,072-token thinking budget*. Whether any of it survives at 8,192 depends on the champion's run (running now).
- Accuracy-when-finished falls 76.8 → 71.3 because harder questions are the ones that need the extra budget.

### 2026-11-09 15:31 — cap-8192 check (D60): result
The champion run finished at 15:31 (about 4 h 50 m; the waiter hit its 2 h limit once and was restarted). `scripts/cap_compare.py`, held-out MMLU-Pro 128, det1, same seeds:

| run | acc | cut | acc when finished | mean tokens |
|---|---|---|---|---|
| base @3,072 | 33.59 | 56.25 | 76.79 | 2,426 |
| champion @3,072 | 45.31 | 46.09 | 84.06 | 2,115 |
| base @8,192 | 56.25 | 21.09 | 71.29 | 4,338 |
| **champion @8,192** | **64.06** | 16.41 | 76.64 | 3,542 |

- **Champion vs base at 8,192: +7.81 pp (SE 2.85, z 2.74), +12/−2.** At 3,072 it was +11.72 (SE 3.06, z 3.83), +16/−1. About two thirds of the gain survives the longer budget.
- **Determinism:** prefix check identical 56/56 (base) and 69/69 (champion). Each model's 8,192 run strictly dominates its 3,072 run (base +29/−0, champion +24/−0).
- **Budget curve.** Accuracy if the cap were T is computed from the 8,192 traces (valid by the prefix check; it reproduces both 3,072 runs exactly):

  | T | 512 | 1,024 | 2,048 | 3,072 | 4,096 | 5,120 | 6,144 | 7,168 | 8,192 |
  |---|---|---|---|---|---|---|---|---|---|
  | base | 3.9 | 13.3 | 24.2 | 33.6 | 41.4 | 46.1 | 50.0 | 51.6 | 56.3 |
  | champion | 10.2 | 21.1 | 36.7 | 45.3 | 51.6 | 59.4 | 62.5 | 62.5 | 64.1 |

  The champion is ahead at every budget. It reaches base's 8,192 accuracy at about 4,800 tokens, and uses 18% fewer tokens on average at 8,192.
- **Where the 8,192 gain comes from.** Only 4 of the 12 gains are questions base still cuts at 8,192. On the 98 questions both models finish, the champion is right on 78 and base on 71. So at the longer budget the edit also changes *what* the model concludes, not only whether it finishes. That is consistent with the L22 two-finalist commit, but it is one paired run; resampling contributes noise both ways.
- GPQA was not re-run at 8,192: 198 questions with 85% cut at 3,072 would take an estimated 10+ h per model. The GPQA transfer figure stays a 3,072-budget figure.

### 2026-11-09 19:20 — generalisation benchmarks (D62) queued
- The model card (LiquidAI/LFM3-1.7B) positions the model for local agents, tool calling, structured output and instruction following.
  - Its own numbers include IFEval 96.7, IFBench 73.0, BFCL V4 71.4 and GPQA Diamond 63.4.
  - These are measured with no output limit; our GPQA figure of 12.6 is at a 3,072 cap.
- Added queue job type `bench` and `wsp/bench.py` (prep / queue / grade).
  - Scorers were unit-tested on hand-made answers: GSM8K number extraction, MC letters, and math-verify on a LaTeX tuple.
- The container had restarted (uptime 10 min). Queue server and watchdog restarted (server pid 701).
- Queued at the 8,192 cap, base then champion: GSM8K (250), MedQA (256), IFEval (541), MATH-500 (200).
- **Blocked:** the session's safety check refused to run the IFEval checker cloned from github.com/google-research (`instruction_following_eval`). IFEval generation proceeds; scoring waits for the user's decision.

### 2026-11-09 21:25 — GSM8K (D62): result
- 250 test questions, 8,192 cap, same per-item seeds for both models. Base took about 1 h, the champion about 1 h.

  | | accuracy | cut | mean tokens | median tokens | visible answer (mean chars) |
  |---|---|---|---|---|---|
  | base | 93.6 | 0.8% (2) | 1,124 | 638 | 518 |
  | champion | 92.8 | 2.0% (5) | 904 | 404 | 53 |

- Paired: −0.8 pp, SE 0.98, z −0.82, +2/−4. **No change in accuracy at 8,192**; GSM8K is near this model's ceiling.
  - Of the 4 losses, 2 are new champion cuts (ids 353 and 901; base finished them at 4,002 and 6,560 tokens) and 2 are wrong finished answers (ids 1046 and 577). Of the 2 gains, 1 is a base cut the champion finishes (id 236) and 1 is a wrong base answer (id 781).
  - On the 244 questions both finish, base is right on 232 and the champion on 231.
- **Budget curve** (accuracy if the cap had been k; valid by prefix determinism, D60):

  | cap | 512 | 1,024 | 1,536 | 2,048 | 3,072 | 4,096 | 8,192 |
  |---|---|---|---|---|---|---|---|
  | base | 40.4 | 64.4 | 77.6 | 84.0 | 88.8 | 91.6 | 93.6 |
  | champion | 58.4 | 74.4 | 83.6 | 89.6 | 90.8 | 91.2 | 92.8 |

  Same shape as MMLU-Pro: a large lead at tight budgets (+18 pp at 512, +10 at 1,024), gone by about 4k.
- The answer-format units carry over to numeric answers: 245/250 champion responses contain "The answer is", and the visible answer shrinks from ~518 to ~53 characters (the L23 short-answer unit). The checker's number extraction handles both.

### 2026-11-10 05:30 — container restart; IFEval base resumed
- At the 05:25 check-in the container had just restarted (uptime 0 min). The last IFEval cache write was 23:28, about 4 min after the last background waiter hit its 2 h limit. That fits the container being reclaimed when the session has no running task, so about 6 h of CPU were lost.
- Restarted the server (pid 206) and watchdog (pid 208). `start_server.sh` moved the running job back to pending, and the job picked up again from the per-item cache (144/541 base answers kept).
- Interim IFEval base, first 137 prompts: 12% cut at 8,192, median 2,744 tokens. That is about 8 h per model, much slower than estimated.
- Mitigation: keep a waiter running in the background whenever runs are queued, and re-check every 2 h.

### 2026-11-10 14:25 — IFEval (D62): base done
- 541 prompts at the 8,192 cap, about 12.5 h of CPU in total (including the restart).
- Official checker results:
  - prompt-level strict 81.7, loose 83.6;
  - instruction-level strict 81.5, loose 82.7.
- Cut 14.4% (78 prompts). Median 3,078 tokens, mean 3,846. Long thinking on simple formatting tasks is the main cost.
- **Finished-only** (463 prompts): strict 95.5, loose 97.6. The model card's 96.66 is with no output limit, so our scoring is calibrated and the shortfall is the 8,192 budget.
- Budget curve (prompt-level strict, if the cap had been k):

  | cap | 1,024 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|
  | base | 9.2 | 29.8 | 47.7 | 61.4 | 73.8 | 81.7 |

- Instruction-level comes out slightly below prompt-level because a cut prompt fails every one of its instructions, and multi-instruction prompts are cut more often.
- Champion run started 14:23.

### 2026-11-10 16:25 — IFEval (D62): champion interim, 108/541
- This is an early-finishing subset, so it is biased toward shorter traces; the final result decides.
- Prompt-level strict: base 84.3 vs champion 88.9 (+10/−5); loose 87.0 vs 88.0. Cut: base 11, champion 9.
- **No sign of format harm so far.** The champion writes full responses (median 854 visible characters vs base 917) and does not collapse to "The answer is (X)".
  - "answer is" appears in 4 champion responses, all natural uses, e.g. a riddle answer and "The short answer is …".
  - The L23 short-answer unit therefore looks conditional on answer-style prompts and does not fire on open-ended writing.
- Of the 5 losses, 3 are champion cuts on prompts where base only just finished (7,751, 7,970 and 8,116 tokens), i.e. noise at the cap boundary. The other 2 are real misses: a truncated-looking JSON (id 13) and a letter-frequency constraint (id 1402).
- Median tokens on this subset are about equal (2,589 vs 2,649). Unlike on MMLU-Pro and GSM8K, the champion does not think less here.

### 2026-11-11 00:50 — IFEval (D62): result
- The champion run took about 10 h; MedQA base started at 00:40.
- **Grading fix:** the official checker is not deterministic.
  - `keywords:letter_frequency` with a non-letter target (e.g. '#', '!', which occurs in the dataset) silently draws a random letter, a known upstream quirk.
  - `change_case:english_*` and `language:*` call an unseeded `langdetect`.
  - Repeated gradings flipped 3 base prompts (ids 1122, 1129, 279), moving accuracy by ±0.2 pp and the paired z between 1.96 and 2.07.
  - `score_ifeval` now seeds `random` per prompt (`ifeval:<key>`) and sets `DetectorFactory.seed = 0`. The checker logic is unchanged; three regrades give identical numbers. The base figure logged at 14:25 (81.7) was an unseeded draw; the seeded figure is 81.9.

  | 541 prompts, 8,192 cap | prompt strict | prompt loose | inst strict | inst loose | cut | mean tokens |
  |---|---|---|---|---|---|---|
  | base | 81.9 | 83.6 | 81.7 | 82.7 | 14.4% | 3,846 |
  | champion | 84.8 | 86.0 | 83.9 | 84.8 | 11.3% | 3,689 |

- Paired, prompt strict: **+2.96 pp, SE 1.43, z 2.07, +38/−22**. Loose: +2.4 pp, z ~1.6. The strict result sits just at the z = 2 line.
- **The gain is all from finishing more often.**
  - 33 of the 38 gains are prompts base cuts; 18 of the 22 losses are prompts the champion cuts.
  - On the 443 prompts both finish, strict is 425 vs 426 and loose 433 vs 430: identical instruction-following quality.
  - Finished-only strict: base 95.7 (n=463), champion 95.6 (n=480).
- **No format harm.** Per-category instruction-level passes on the both-finish set are equal within ±1 in every category, including detectable_format at 136/136 for both and length_constraints at 104 vs 103. The visible response stays full length (median 880 vs 856 characters). "answer is" appears in 11 base and 17 champion responses, all natural uses.
- Budget curve (prompt strict):

  | cap | 1,024 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|
  | base | 9.2 | 29.8 | 47.7 | 61.4 | 73.8 | 81.9 |
  | champion | 9.8 | 30.7 | 52.1 | 63.0 | 75.4 | 84.8 |

  The lead is small everywhere, at most +4.4 pp at 3,072. That is much smaller than on MMLU-Pro and GSM8K, where the tight-budget gains were +10 to +18 pp: the wrap-up units barely shorten open-ended writing (median 3,078 → 2,943 tokens).

### 2026-11-11 02:25 — second container restart; MedQA base resumed
- The container restarted at about 02:23 while a background waiter was still running, so this one was not the idle reclaim of 2026-11-10.
- Cache intact: 79/256 MedQA base answers. Server (pid 191) and watchdog (pid 193) restarted, and the job went back to pending and resumes from the cache. Only the in-flight batch was lost.

### 2026-11-11 06:40 — MedQA (D62): base done
- 256 USMLE 4-option questions at the 8,192 cap, about 6 h. Champion run started 06:37.
- **Accuracy 55.9** (SE 3.1). Cut 19.9%, mean 4,511 tokens. Finished-only: 69.8 (n = 205). Every finished trace gave a parseable letter.
- Budget curve:

  | cap | 1,024 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|
  | base | 5.9 | 21.5 | 32.0 | 39.5 | 46.5 | 55.9 |

  The shape is very close to MMLU-Pro base (33.6 at 3,072, 56.3 at 8,192). Medical questions are as thinking-hungry as general MMLU-Pro for this model.

### 2026-11-11 12:20 — MedQA (D62): result
- The champion run took about 5.5 h (restart at 06:38). MATH-500 base started at 12:13.

  | 256 questions | accuracy | cut | mean tokens | median tokens | finished-only |
  |---|---|---|---|---|---|
  | base | 55.9 | 19.9% | 4,511 | 4,027 | 69.8 (n=205) |
  | champion | 55.9 | 16.4% | 3,849 | 3,164 | 66.8 (n=214) |

- **Paired at 8,192: 0.0 pp, SE 2.54, +21/−21.** No change.
  - 16 of the 21 gains are base cuts; 12 of the 21 losses are champion cuts.
  - On the 185 questions both finish: base 131, champion 127.
  - Every finished champion trace gave a parseable letter.
- **Budget curve:**

  | cap | 1,024 | 2,048 | 3,072 | 4,096 | 5,120 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|---|
  | base | 5.9 | 21.5 | 32.0 | 39.5 | 42.6 | 46.5 | 55.9 |
  | champion | 10.9 | 27.7 | 38.7 | 44.9 | 48.4 | 53.5 | 55.9 |

  At the selection budget, 3,072: **+6.6 pp, z 3.21, +23/−6**. The champion leads at every cap up to 6,144 and ties at 8,192. Median tokens fall 21%.
- **Contrast with MMLU-Pro:**
  - On MMLU-Pro at 8,192 the champion was also better on the questions both models finish (78 vs 71), which we read as the L22 finalist-commit unit changing conclusions.
  - On MedQA it is slightly worse on that set (127 vs 131; −4 out of 185, within noise).
  - So the efficiency effect transfers to medicine, but the content effect seen on MMLU-Pro does not, or is MMLU-Pro-specific.

### 2026-11-11 15:00 — MATH-500 (D62): base done
- 200 sampled problems at the 8,192 cap, about 2.8 h. Champion run started 14:59.
- **Accuracy 85.5** (SE 2.5). Cut 14.5% (29), mean 2,533 tokens.
- **Every one of the 171 finished traces is correct.** All misses are cuts, mostly hard ones (level 5: 16, level 4: 8, level 3: 4, level 2: 1).
- Grader check:
  - math-verify accepted none of 60 deliberately wrong boxed answers;
  - it accepts equivalent forms, e.g. `(3,\frac{\pi}{2})` for `\left( 3, \frac{\pi}{2} \right)`.
  - The 100% is therefore real. Possible causes are that the model is strong at MATH or that MATH-500 is in its training data; we cannot tell which.
- So at this budget MATH-500 measures only whether the model *finishes*.
- Budget curve:

  | cap | 1,024 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|
  | base | 48.5 | 65.5 | 70.5 | 76.5 | 84.0 | 85.5 |

### 2026-11-11 17:45 — MATH-500 (D62): result
- The champion run took about 2.7 h. The queue is now empty.

  | 200 problems | accuracy | cut | mean tokens | median tokens | finished-only |
  |---|---|---|---|---|---|
  | base | 85.5 | 14.5% | 2,533 | 1,064 | 100.0 (171/171) |
  | champion | 87.0 | 12.5% | 2,447 | 1,041 | 99.4 (174/175) |

- Paired at 8,192: **+1.5 pp, SE 1.94, z 0.77, +9/−6**. Not significant.
  - All 9 gains are base cuts; 5 of the 6 losses are champion cuts.
- The champion's single "wrong" finished answer is a **grader false negative**: a correct `pmatrix` written with `\\[4pt]` row spacing, which math-verify fails to parse. Counted as correct, the champion would be 87.5 (+2.0 pp). We report the grader's number.
- Budget curve, paired diff (z):

  | cap | 512 | 1,024 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|---|
  | base | 21.0 | 48.5 | 65.5 | 70.5 | 76.5 | 84.0 | 85.5 |
  | champion | 25.5 | 48.5 | 67.5 | 72.0 | 76.5 | 82.0 | 87.0 |
  | diff | +4.5 (2.4) | 0 | +2.0 (1.1) | +1.5 (0.9) | 0 | −2.0 (−1.2) | +1.5 (0.8) |

- **No efficiency effect on MATH-500:** median tokens are 1,064 vs 1,041. This contrasts with GSM8K (−37%), MMLU-Pro and MedQA (−21%).
  - Hypothesis H62a: the wrap-up and short-answer units key on the answer-format instruction in the prompt ("finish with 'The answer is …'"), which GSM8K, MMLU-Pro and MedQA share and MATH-500 (`\boxed{}`) and IFEval lack. IFEval also showed little token saving (−4%).
  - A direct test would be MATH-500 with the "The answer is" instruction, or GSM8K with `\boxed{}`. Not run.

### 2026-11-11 17:55 — H62a test queued: GSM8K with `\boxed{}`
- Same 250 GSM8K questions, ids and seeds as `gsm8k`. The only change is the instruction: "put your final answer within \boxed{}" instead of "finish with 'The answer is N'". Graded with math-verify, unit-checked: `\boxed{1,200}` matches 1200, and 19 does not match 18.
- Prediction if H62a holds: the champion's token saving (−37% median with the "answer is" instruction) largely disappears, and the 512-token lead (+18 pp) shrinks.
- If the saving persists, the units respond to the task (short arithmetic) rather than to the instruction format.
- About 2 h of CPU in total.

### 2026-11-11 — handoff tooling; H62a interim
- `scripts/champion_delta.py` (D64) and `scripts/verify_engine.py` were added so the project can continue on a new VM.
  - `verify_engine.py N` regenerates the N shortest finished base traces of the committed det1 bank on port 8092 and compares them byte for byte.
  - Test here: **4/4 byte-identical**. It took 24 min because it shared 4 cores with a running queue job; run it before starting the server.
- **H62a interim** (GSM8K with `\boxed{}`, first 25 champion answers, a short-biased subset), median tokens on the same questions:

  | instruction | base | champion | saving |
  |---|---|---|---|
  | "The answer is N" | 503 | 326 | −35% |
  | `\boxed{}` | 484 | 450 | −7% |

  - With `\boxed{}` the visible answer stays full length (median 462 vs 446 characters).
  - Early support for H62a: both the short-answer and the wrap-up effects are largely conditional on the "The answer is" instruction.
- Base with `\boxed{}`: 10/250 cut at 8,192, against 2/250 with "The answer is N". The instruction format alone changes base behaviour.

### 2026-11-11 — H62a result: the effect is conditional on the "The answer is …" instruction (confirmed)
- GSM8K with `\boxed{}`, 250 questions, cap 8,192, both models complete.
  - Accuracy: base 92.0, champion 91.2. Paired diff −0.8 (SE 0.8, z −1.0, +1/−3). Cuts: 10 vs 9.
- On questions both models finish, the token savings split into thinking and visible-answer parts (median characters; `tokens` counts thinking and answer together):

  | run | both-finish | thinking chars | visible answer chars | total tokens |
  |---|---|---|---|---|
  | GSM8K "The answer is N" | 244 | 1,592 → 1,276 (−20%) | 482 → 18 | 622 → 395 (−36%) |
  | GSM8K `\boxed{}` | 238 | 1,061 → 1,061 (0%) | 453 → 455 | 488 → 494 (+1%) |
  | MedQA "The answer is (X)" | 185 | 11,199 → 9,482 (−15%) | 1,118 → 19 | 2,806 → 2,126 (−24%) |
  | MATH-500 `\boxed{}` | 166 | 1,480 → 1,414 (−4%) | 688 → 635 | 798 → 752 |
  | IFEval (free-form) | 443 | 9,747 → 9,331 (−4%) | 880 → 856 | 2,574 → 2,443 |

  - Median per-question token ratio on GSM8K-boxed is 0.99.
- Accuracy by budget (only answers that finished within the budget count):

  | budget | 256 | 512 | 768 | 1,024 | 1,536 | 2,048 | 3,072 | 4,096 | 8,192 |
  |---|---|---|---|---|---|---|---|---|---|
  | base, answer-is | 9.2 | 40.8 | 55.6 | 64.4 | 77.6 | 84.0 | 88.8 | 91.6 | 93.6 |
  | champion, answer-is | 32.0 | 58.8 | 70.0 | 74.4 | 83.6 | 89.6 | 90.8 | 91.2 | 92.8 |
  | base, boxed | 14.8 | 49.6 | 62.8 | 70.0 | 81.2 | 85.6 | 88.0 | 89.6 | 92.0 |
  | champion, boxed | 13.6 | 49.6 | 66.0 | 73.2 | 80.8 | 84.4 | 88.8 | 89.2 | 91.2 |

- **H62a confirmed.** Both units are conditional on the "The answer is …" instruction:
  - the short-answer unit (no terse answers with `\boxed{}`);
  - the wrap-up unit (no thinking saving with `\boxed{}`).
  - On the same arithmetic questions with a different answer instruction, the champion behaves like the base model.
- **Correction to earlier entries.** The "median-token change" figures (GSM8K −37%, MedQA −21%) are total generated tokens. Roughly half of the GSM8K saving and a third of the MedQA saving come from the one-line visible answer; the thinking itself shrinks 20% and 15%.
- **Base-model observation.** The "The answer is N" instruction itself makes the *base* model think longer than `\boxed{}` (1,592 vs 1,061 thinking chars) and do worse at small budgets (40.8 vs 49.6 at 512).
  - Part of the champion's GSM8K tight-budget lead therefore repairs a cost that this prompt format imposes on the base model.
  - The champion with "The answer is" still beats the base with `\boxed{}` by 4–9 pp at 512–2,048 tokens, and ties above that.

### 2026-11-11 23:40 — Generation 10 set up for reasoning (content) gains (D65)
- User direction: "Continue aiming for intelligence". The selection size and the `\boxed{}` guard were left to Claude: 256 questions at 8,192 tokens, a both-finish content check, and a GSM8K-boxed harm guard (D65).
- Container restart at about 23:37 (fifth). Restarted the server (pid 278) and the watchdog (pid 288).
- Queued (parent = gen-8 champion, cap 8,192):
  1. Headroom: 64 trace-bank ids × 4 samples (`runs/shared/passk_gen008champ_cap8192_det1.jsonl`), priority 3, about 4 h.
  2. The 48-id holdout bank and the 96-id trace bank at 8,192, priority 4.
- Code: `ga new-gen --sel-cap`, `ga select --parent-chunks`, `ga guard`, `decide --bf-z/--guard-z`, both-finish statistics; editor generate passes accept `"cap": 8192`. The server was re-executed between jobs to load them.
- Check on old data: generation 9's two candidates had **0 changed answers** on the 71 and 70 selection questions both models finished at 3,072 (+0/−0). Their whole selection effect was finishing, or not finishing, traces.
- Brief `prompts/editor_task_think_g10.md`, drafted from g9: content-failure targets, the new selection rule, 8,192 banks, generation-9 and D60–D63/H62a findings. The headroom numbers are filled in when the run finishes.
