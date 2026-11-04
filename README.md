# Weight-space programming, as a genetic loop

A replication of Experiment 8 of *Sparks of Weight-Space Programming*
(https://rikiyat.github.io/blog/weight-space-programming/, logs: github.com/RikiyaT/lm_understand-logs), turned into a generational loop.

| | Paper (Exp 8) | Here |
|---|---|---|
| Edited model | Qwen3.5-0.8B, thinking on | **LFM3-1.7B** (Liquid AI, Oct 2026), direct answer (thinking off) |
| Editor | GPT-6 Astra (Codex CLI, tools off) | Claude (this session) + Claude sub-agents, one per editor slot |
| Target | MATH-500 / GSM8K (generation, 32k tokens, vLLM on GPU) | **MMLU-Pro** (letter logits after "The answer is (", CPU) |
| Practice data | GSM8K / MATH train | 3,007 MMLU-Pro questions (25%, stratified by subject) |
| Kernel API | load / load_original / save_edit / drop_edit / edited / tok | the same + load_base, submit_candidate, render, LETTER_IDS |
| Forward tool | generate ≤512 (both ≤256/model) or teacher-forced ≤1,024 texts; budget 10/25/50 | score ≤512 (both ≤256/model) or read ≤1,024 texts; budget per editor per generation |
| Rules | no training, no running the model in the kernel, no embedding/lm_head/final-norm edits, no whole-tensor scaling | identical (enforced in save_edit + stated in the brief) |
| Loop | one run | **generations**: N editors -> ≤ N×M candidates -> hidden paired selection -> winner (z ≥ 2) becomes parent; else parent survives |

## Splits (MMLU-Pro test, 12,032 questions, stratified per subject, fixed seeds)
- `practice` 3,007: editors see these with answers.
- `selection` pool 2,046: hidden; each generation samples its own subset to choose winners (limits overfitting to one set).
- `heldout` 6,979: untouched; used only for checkpoint and final evaluations. GPQA Diamond (198) is used the same way.

## Layout
```
wsp/config.py      constants, forbidden tensors, tool limits
wsp/data.py        splits, the fixed evaluation prompt, GPQA builder
wsp/engine.py      the model (bf16, CPU): weight swapping, scoring, activation capture
wsp/server.py      the forward-pass server (one queue for all editors; budget accounting)
wsp/fwd.py         the editor's forward-pass client
wsp/editor.py      the editor API (rules enforced here)
wsp/ga.py          the GA driver
prompts/editor_task.md   the editor's brief (adapted from the paper's task message)
runs/              ga_state.json, gen_NNN/ (editors' folders, selection results, decision), champions/, heldout/
queue/             the server's queue, heartbeat and log
```

## Running a generation
```
./start_server.sh                                   # once (restart if the heartbeat is stale)
python -m wsp.ga new-gen --editors claude,ed_a,ed_b --budget 10 --max-cands 2 --sel-n 600
python -m wsp.ga brief                              # writes TASK.md into every editor folder
#   ... editors work (each reads its TASK.md, uses <edir>/py and <edir>/fwd, submits candidates) ...
python -m wsp.ga select --wait                      # hidden paired evaluation of all candidates
python -m wsp.ga decide --z 2.0                     # winner -> new champion (or parent survives)
python -m wsp.ga heldout champion --n 1000          # checkpoint evaluation
```
