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
