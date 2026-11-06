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
