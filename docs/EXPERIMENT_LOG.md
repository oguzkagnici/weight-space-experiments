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
