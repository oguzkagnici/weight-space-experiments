# Experiment brief: writing a ≤ 2B language model's weights directly, without training

You are the **experiment runner** (the orchestrator). This document is your complete brief. Read it once fully, then start with **§13, First actions**. Make every choice it leaves open yourself, log it (§10), and keep going without waiting for approval unless §12 says to stop and ask.

---

## 1. The question

Can an LLM, acting as an engineer who **writes model weights as numbers** with Python, create a language model of **at most 2.0 billion parameters** that is **generally intelligent** (MMLU-Pro, GPQA Diamond and similar benchmarks), using:

- **no training:** no gradient steps, no optimizer, no fitting of weights to a loss;
- **no training data:** no corpus, no distillation, no instruction data.

**The goal is to make an LLM by writing its weights, not to mix existing models.** The runner and its editors decide the architecture and write the tensors themselves: from their own understanding of how transformers compute, as analytic constructions, designed circuits, chosen structures, or any numbers they decide to write.

**Existing open models (the reference models) are a resource, not the product.** The runner may:
- **study** them, to learn how real networks implement things (weight statistics, circuits, embeddings, norms, attention patterns), and then write its own weights informed by that;
- **use parts** of them where writing a part is out of reach (e.g. take one model's embedding table, or one block), or mix them.

This is allowed and expected where it helps. But every candidate's **provenance** (§8) states how much of it was written and how much was taken. The experiment's headline is both numbers together: **how intelligent is the model, and how much of it was written**. A model that is mostly a copy or a merge of references scores what its references score, and says nothing about writing an LLM. It is a reference point, not a result.

**Be honest about the framing in every report.**
- Written weights that encode knowledge come from the editor LLM's own understanding. That understanding itself came from its training, and the reference models' knowledge came from theirs.
- "From scratch" here means **built by direct weight writing, not by training**.
- Report clearly which parts of the final model were written and which were taken.

This experiment follows a predecessor in which LLM editors directly edited one 1.7B model's weights in a genetic loop. Its lessons are in §11. Use them.

---

## 2. Settings (defaults; change only through a logged decision)

| Setting | Default | Notes |
|---|---|---|
| `MAX_PARAMS` | 2.0 × 10⁹ | Count all parameters, embeddings included; tied tensors count once. |
| `REF_SIZE_RANGE` | 0.3B – 4B total params | Models to study and optionally take from. Larger ones may be studied or used as material (e.g. pruned). |
| `REF_BAR` | best **eligible reference model** (≤ `MAX_PARAMS`) on the sealed final composite | The comparison point for intelligence, not the goal (§7.5). |
| `LINEAGES` | 2: **W** (written) and **M** (main) | §7.3. |
| `W_MIN_AUTHORED` | 0.90 | Lineage W: share of parameters that must be authored (§8). |
| `AUTHORING_STEP` | +2 percentage points of total params | Minimum authored-share gain for an "authoring move" in lineage M (§7.3). |
| `EDITORS` | 3 per generation: 2 on lineage M, 1 on lineage W | Use different underlying models if the environment allows it (e.g. different Claude models). |
| `PASSES_PER_EDITOR` | 10 forward-tool requests per generation | |
| `CANDIDATES_PER_EDITOR` | ≤ 2 | |
| `GENERATIONS` | 15, or stop on stagnation (§7.5) | |
| `PROTOCOL` | reasoning allowed, `CAP` new tokens, temperature 0.6, top-p 0.95, top-k 20 | §5. |
| `CAP` | 8,192 tokens with a GPU; 2,048 on CPU only (§5.6) | |
| `SEL_SIZES` | MMLU-Pro 512, GPQA 250, GSM8K 400 | Hidden selection sets (§4). |
| `SELECT_Z` | 2.0 on the composite; each primary benchmark z ≥ −1; GSM8K guard z ≥ −2 | §7.3. |

---

## 3. Reference models

### 3.1 Census (Phase 1)

Build `docs/REFERENCES.md` with one row per reference model. **Verify everything on the Hub at the time you run.** New models appear monthly, and the list below is only a starting point from the brief's author:

- `Qwen/Qwen2.5-1.5B` (base) and its Instruct, Math and Coder variants; `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- `Qwen/Qwen3-1.7B` and `Qwen3-1.7B-Base`; `Qwen3-0.6B`; `Qwen/Qwen3-4B` (study / material only).
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` and `SmolLM2-135M/360M` (small models are easy to study whole); `SmolLM3-3B`.
- `meta-llama/Llama-3.2-1B-Instruct` (gated); `google/gemma-3-1b-it` (gated).
- `allenai/OLMo-2-0425-1B-Instruct` (open training data and intermediate checkpoints: useful for studying how weights look).
- `ibm-granite` 2B-class models; `LiquidAI` LFM2/LFM3 small models (non-transformer hybrid).
- Any newer ≤ 4B open model with a usable license.

For each reference record:
- exact repo and revision hash;
- license (refuse models whose license forbids derivatives or redistribution, and record the reason);
- total and non-embedding parameter counts;
- architecture class and its key dims: layers, d_model, heads, kv-heads, head_dim, ff, activation, norm type, qk-norm, biases, RoPE theta, sliding windows, tied embeddings;
- tokenizer: type, vocab size, special tokens, chat template, thinking support;
- known training-data claims.

**Exclude any reference whose card claims training on MMLU-Pro, GPQA, GSM8K-test or MATH-500 test items.**

### 3.2 Study material for the editors

Put in `docs/REFERENCES.md` what helps editors **write** weights, not only copy them:
- weight-space statistics per tensor type: norms, spectra (top singular values), value distributions, per-layer trends;
- cross-model comparisons: per-layer cosine similarity of same-shape tensors; which structures recur across families (e.g. norm scales, attention-sink heads, positional heads);
- a compatibility map for editors who choose to take parts: vocab overlap by shared token strings, dimension mismatches, norm placement, RoPE differences.

### 3.3 Downloads and secrets

Gated models and GPQA need an HF token supplied by the user:
- store it in a file outside the repo;
- never print, log or commit it;
- add a pre-commit check that refuses staged files containing `hf_`.

Keep model weights out of git (`.gitignore`). Commit only manifests, recipes and hashes.

---

## 4. Benchmarks and data hygiene (most important section)

### 4.1 Sets

| Benchmark | Source | Practice (editors see, with answers) | Selection (hidden; decides each generation) | Final (sealed; run once at the end) |
|---|---|---|---|---|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro`, test split (12,032 q, ≤ 10 options, 14 subjects) | 3,000 | 512 | 2,000 |
| GPQA | `Idavidrein/gpqa` (gated); `gpqa_extended` ⊃ `gpqa_main` ⊃ `gpqa_diamond` | extended − main (≈ 98) | main − diamond (≈ 250) | **Diamond (198)** |
| GSM8K | `openai/gsm8k` `main` | the train split (7,473) | 400 of test (guard) | the other 919 of test |
| MATH-500 | `HuggingFaceH4/MATH-500` | none | none | all 500 (secondary, final only) |

- The remaining ~6,500 MMLU-Pro test questions form a **reserve**: never used, kept for a fresh final set if the experiment is extended after the final run.
- Verify the GPQA nesting by Record ID before splitting and log the counts.
- Splits are **stratified by subject**, using a seeded hash order: sort by `sha256(f"{SPLIT_SEED}:{dataset}:{id}")`. Draw `SPLIT_SEED` once and log it.
- GPQA options are shuffled once per question with seed `gpqa-<record id>`. Log the resulting answer-letter distribution; it should be roughly uniform.
- The headline metric is **MMLU-Pro final + GPQA Diamond**, reported separately and as their mean, always next to the model's authored share (§8). GSM8K-final and MATH-500 are secondary.
- Chance levels: MMLU-Pro ≈ 10% (up to 10 options), GPQA 25%. Report them; early written models will be near chance.

### 4.2 Who may see what

| Party | Practice | Selection | Final |
|---|---|---|---|
| Editor sub-agents | full text and answers | **nothing**: no ids, no files, no per-question results | **nothing** |
| Orchestrator (you) | full | ids and per-question **correctness only**, for statistics; **never** read or print question text, options or model outputs for these items | ids only; aggregate numbers at the very end |

### 4.3 Enforcement

- **Files.**
  - Hidden and final question files live in `data/sealed/`.
  - The evaluation harness reads them; your analysis scripts read only `results/*.jsonl` rows reduced to `{id, correct, cut, tokens}`.
  - Write a helper that strips everything else, and use only that helper.
- **Commitment.** Before any model is run, write the sha256 of each sorted id list (practice, selection and final, per benchmark) into `docs/DECISIONS.md`.
- **Editor folders contain:**
  - practice data;
  - their own work;
  - the API (§6).
- **Editor prompts say:** do not read outside your folder, the API and the shared read-only reference directory. In particular, never read `data/sealed/`, `results/`, the GA state, queue files or other editors' folders.
- **No benchmark content in weights.** Editors must not write benchmark questions, answers or answer keys into weights (e.g. an embedding or MLP that stores "question X → letter C"). Practice items are for measuring, not for storing. The provenance audit (§8) flags authored tensors that correlate with practice-item answers; a flagged candidate is rejected.
- **Final set timing.**
  - It runs exactly once, after the last generation, on both lineage champions, every eligible reference model and the baselines of §7.1, all in one batch.
  - One optional mid-experiment final run is allowed only if you pre-register it, with its generation number, in DECISIONS before generation 1.
  - If the experiment continues after a final run, build a new final set from the reserve.
- **Copyright and the GPQA request.** Do not commit benchmark question text. GPQA asks that its examples not be revealed online. Commit ids and hashes only.
- **Contamination you cannot control.** The editor LLMs and the reference models may have seen these benchmarks in pretraining. Note this in the final report.

---

## 5. Evaluation protocol (identical for every model)

1. **Engine.**
   - With a GPU: vLLM, or transformers with batched generation.
   - CPU only: llama.cpp (convert candidates to GGUF Q8_0).
   - Pin versions and log them.
   - **Architecture freedom has a practical limit:** a candidate must run in the engine. Prefer standard architectures the engine supports (Llama/Qwen-style decoder transformers, with any dims, layer counts, head counts, norms or activations they allow). A custom architecture is allowed only if you implement and test it in the engine first, by a logged decision.
2. **Prompt.** One fixed user-message template per benchmark, wrapped in **the model's own chat template**:
   - multiple choice: the question, then `(A) …` option lines, then "Think step by step, then finish with: The answer is (X).";
   - GSM8K / MATH: "… finish with: The answer is \boxed{…}."
   - Reference models that support thinking run with thinking on.
   - A created model may use any tokenizer and chat template, written by the editors or taken from a reference. It is fixed per lineage in generation 0 and changes only by a logged decision.
3. **Sampling.**
   - Temperature 0.6, top-p 0.95, top-k 20, max `CAP` new tokens.
   - Seed per item: `crc32(f"{benchmark}:{id}:{sample}")`. Sample 0 is the default; GPQA uses 4 samples per question (8 at the final).
4. **Extraction.**
   - The last "answer is (X)", else the last `\boxed{}`, else unparsable (wrong).
   - Mark `cut = True` when generation hit `CAP`; a cut answer scores wrong.
   - Store `{id, sample, pred, gold, correct, cut, tokens, raw}`, where `raw` is the full output.
   - **Format-only scores.** A model that outputs "The answer is (A)" every time scores ~10% on MMLU-Pro without any intelligence. Report each model's answer-letter distribution and its accuracy against a letter-prior baseline (always the model's most frequent letter). Near-chance written models must be read against this baseline, not against 0.
5. **Determinism check (Phase 0).**
   - Run one reference model twice on 32 practice items and check that outputs are byte-identical.
   - If the engine is not deterministic (batch effects), log it and rely on paired statistics with the resampling control below. Never on luck.
   - Cache results by `(weights hash, benchmark, id, sample, cap)`.
   - Every time you compare two models, check that their weight hashes differ. Count byte-identical outputs: identical outputs between different weights are expected only where the change is tiny.
6. **Throughput first.**
   - Measure tokens/s and time per answer at `CAP` before fixing set sizes.
   - If a full selection of one candidate (`SEL_SIZES`) takes more than ~2 hours, shrink the sets or `CAP` and log why.
   - On CPU only, use `CAP` 2,048, MMLU-Pro selection 256, GPQA selection 250 × 1 sample, and say in DECISIONS that results are budget-bound.
   - Written models that are not yet coherent may ramble to `CAP` on every item. Screen them with `ppl` and a short `generate` first (§6) and skip full selection for candidates that cannot produce the answer format.
7. **Statistics.**
   - Always **paired**: per-item difference (candidate − reference), mean, SE = sd/√n, z = mean/SE, and counts +/−.
   - Also report the **both-finish** subset (items neither model cut).
   - For GPQA with k samples, average per question first, then pair by question.

---

## 6. Editors and the tool API you must build (Phase 0)

Editors are sub-agents. Each generation, each editor works in its own folder with this API. Build it as a small Python package plus a queue-backed server; adapt the predecessor's design:

- **Read-only references.** `refs()` lists them; `load(ref, tensor_name)` returns a float32 tensor. Also provide `config(ref)`, `tokenizer(ref)` and `names_shapes(ref)`. Every `load` call is recorded for provenance (§8).
- **Candidates are recipes.** An editor writes `recipe.py`: a deterministic Python program that writes every tensor of the model, plus its config, tokenizer and chat template.
  - The recipe is organised in **tensor groups**. Each group is a function that returns its tensors, e.g. `def layer_3_attention(ref): ...`.
  - A group may write numbers directly (formulas, constructions, hand-designed values, seeded random draws), may read reference tensors through `ref.load`, or both.
  - `build(recipe)` runs each group with its own counter of reference reads. It produces a safetensors directory and reports:
    - the parameter count;
    - the weights hash;
    - the provenance (§8).

  Recipes keep every candidate reproducible and auditable. Weights can be deleted and rebuilt, which saves disk (a 2B model is ~4 GB in bf16).
- **Forward tool** `fwd <request.json>`: one request = one pass from the budget. All requests use **practice items only**.
  - `score`: direct-answer letter log-probs on ≤ 512 practice MMLU-Pro/GPQA items. Fast. Optionally saves activations at the answer position.
  - `read`: teacher-forced texts (≤ 1,024 texts, ≤ 200k tokens). Returns token log-probs and top-k; optionally hidden states, MLP inputs and attention from the last token, at chosen layers. Editors use it to test whether a written circuit does what it was designed to do.
  - `generate`: the §5 protocol on ≤ 48 practice items (≤ 24 at `CAP` 8,192), with `"model": "candidate" | "parent" | "both"`. Parent results are cached.
  - `ppl`: perplexity on short, fixed, non-benchmark text, formatted with the model's own chat template. It shows whether a candidate is a working language model at all, which matters most in lineage W.

  A request that errors does not count.
- **Queue.** One server and one queue for everyone, with priorities:
  1. editor passes;
  2. selection;
  3. controls.

  Jobs survive restarts; a watchdog re-queues running jobs after a crash. The container may restart, so make everything resumable from files.
- **Submission.** `submit_candidate(label, recipe, description, lineage)`, at most `CANDIDATES_PER_EDITOR`.
  - The description states what was written and why (the mechanism or design), what was taken from references and why it could not be written, and the measured paired result on the editor's own held-back practice items.
  - Every editor writes `report.md`, which the next generation's editors read.

### 6.1 Rules for editors (put them verbatim in every brief)

**The aim.** You are writing a language model's weights. Prefer writing a part over copying it. Where you take a part from a reference, say why writing it was out of reach, and look for parts of the model you can replace with written weights without losing accuracy.

**Not allowed:**
- (a) Gradient computation of any kind: no backward passes, autograd or optimizers. No fitting of weights to a loss on data, iterative or in closed form; for example, no least-squares solving of a matrix to reproduce recorded outputs, and no distillation.
- (b) Running a model other than through the forward tool. Do not implement or simulate the network's forward computation on inputs in Python. Reading recorded activations through an output matrix (logit lens) is allowed.
- (c) Internet access, and any search for benchmark answers.
- (d) Reading outside your folder, the API and the reference directory.
- (e) Automated search loops that score more than ~20 configurations, such as grid searches over coefficients. Every candidate must be a choice you reasoned about.
- (f) Storing benchmark items or answers in weights (§4.3), and laundering reference weights: a group that hard-codes more than 1,000 numbers copied from a reference (pasted arrays, values read earlier and written back as literals) counts as taken, not written, and must read them through `ref.load` so provenance shows it.

**Allowed:**
- writing any numbers: analytic constructions (e.g. an attention head built to copy the previous token, an embedding with designed geometry, an MLP that implements a lookup you designed), structured matrices, seeded random draws, norms and biases set by reasoning;
- any analysis of reference weights, and writing your own weights informed by what you learned;
- any deterministic construction from reference weights: copy, average, interpolate, task arithmetic, permutation, rotation, projection, SVD truncation, layer splicing, width or depth changes, vocab remapping, pruning;
- statistics of **recorded activations on practice items**, used as means, differences, principal components or correlation-based matching.

**Disclosure.** Every candidate's metadata must state whether it used activation statistics (`uses_activation_stats: true/false`). The final report separates models whose written parts are fully data-free from the rest.

---

## 7. The loop

### 7.1 Baselines (Phase 1; selection sets only — final sets wait until the end)

- **B-ref:** every reference model ≤ 4B on the practice sets (visible) and the selection sets (aggregates only). Eligible references (≤ `MAX_PARAMS`) define the working intelligence bar.
- **B-random:** a standard architecture with seeded random weights (standard initialisation). This is the "nothing written" floor, at about chance.
- **B-merge:** up to 3 standard, mechanical merge recipes you write before seeing any selection result (e.g. a uniform average of same-family fine-tunes; task arithmetic over a shared base). They show what plain mixing gives, so the report can separate it from what writing adds.
- **Resampling control:** the best eligible reference at sample 1 vs sample 0 on selection. This measures the noise floor: how many answers flip with only a seed change.

### 7.2 Generation 0: first models

- Each editor builds ≤ 2 initial models within `MAX_PARAMS`:
  - **at least one for lineage W** (authored share ≥ `W_MIN_AUTHORED`): the editor chooses the architecture and tokenizer and writes the weights;
  - optionally one for lineage M (any authored share).
- The editors receive §3's census and study material, the B-ref practice results and the B-random floor.
- Suggest directions, but do not prescribe them:
  - for W: a small, well-understood architecture; written embeddings with a designed geometry; induction and previous-token heads written analytically; MLP lookups for simple facts or arithmetic; norms set from the arithmetic of the residual stream. Aim first for a model that produces coherent text and the answer format, then for above-chance answers;
  - for M: start from reference parts where needed, but write every part you can (e.g. written attention circuits in a reference body, a written tokenizer and embedding with a reference body, written norms, written layers inserted or replacing reference layers);
  - **for both: replace before you add.** Find which reference parts can be swapped for written ones without a measurable loss; the authored share is the experiment's second axis.
- The champion of each lineage is its candidate with the best selection composite, **even if it is near chance (W) or below the best reference (M)**.

### 7.3 Each generation

1. **Brief.** Write each editor's brief: this experiment's purpose, the rules (§6.1), the API, the history (all previous reports and what was learned), its lineage's parent recipe, provenance and practice results, and a **different suggested starting point** per editor, for diversity.
2. **Launch** editors in parallel, as background sub-agents with their own folders: 2 on lineage M, 1 on lineage W (adjust by a logged decision, e.g. when one lineage stagnates).
3. **Pre-run selection.** As each candidate arrives, build it, check `MAX_PARAMS`, provenance and the audit (§8), screen it (`ppl`, answer format), and pre-run its selection in chunks, at lower priority than editor passes, so editors are not blocked.
4. **Decide.** For each candidate, compute paired diffs vs its lineage's parent on MMLU-Pro-sel and GPQA-sel:
   - D = (d_MMLU + d_GPQA)/2;
   - SE = √(SE_MMLU² + SE_GPQA²)/2;
   - ΔA = the change in authored share (percentage points of total params).

   **Lineage M.** A candidate is promoted by either kind of move:
   - **intelligence move:** D/SE ≥ 2; each benchmark z ≥ −1; GSM8K-sel guard z ≥ −2; ΔA ≥ −1 point (it may not buy accuracy by taking back much written material);
   - **authoring move:** ΔA ≥ `AUTHORING_STEP`; D/SE ≥ −1; each benchmark z ≥ −1.5; GSM8K guard z ≥ −2 (it writes more of the model without a significant loss).

   If several pass, an intelligence move beats an authoring move. Among intelligence moves, the largest D wins; among authoring moves, the largest ΔA. Otherwise the parent survives (elitism).

   **Lineage W.** Promoted on an intelligence move with authored share ≥ `W_MIN_AUTHORED`. While its champion is below the letter-prior baseline (§5.4), use the language-model screen as a tie-breaker: lower `ppl` and a higher rate of well-formed answers, both paired and logged.
5. **Log** the decision (§10) with a table:
   - candidate and lineage;
   - authored share and ΔA;
   - per-benchmark diff (SE), z, +/−;
   - both-finish diff;
   - identical-output count;
   - verdict and move type.

   Commit.

### 7.4 Pre-registration

Before reading a candidate's selection result, write one line in EXPERIMENT_LOG: your prediction and the check you will run (e.g. "the gain should come from category X"; "replacing these heads should cost nothing"). Afterwards, report the check. This keeps the analysis honest.

### 7.5 Stopping and the final report

Stop after `GENERATIONS`, or after 4 consecutive generations without promotion in both lineages, or on user request. Then:

1. **Final run (once):** both lineage champions, every eligible reference model, B-random, the B-merge winner, and the gen-0 champions, on MMLU-Pro-final, GPQA Diamond (8 samples), GSM8K-final and MATH-500.
2. **Report:**
   - accuracy with 95% CIs, next to each model's authored share;
   - a plot of final composite accuracy against authored share, with the references at 0%, B-random, every champion of the run, and chance and letter-prior lines;
   - paired diffs vs the best eligible reference and vs B-random;
   - the answers to the experiment's questions, each with its evidence:
     1. **How intelligent a model can be written** (lineage W): its composite vs chance and vs B-random.
     2. **How much of a strong model can be written** (lineage M): the authored share reached at no significant loss vs the best reference (paired z ≥ −1), and which parts were written.
     3. **Whether writing adds intelligence:** the composite paired z ≥ 2 vs the best eligible reference, by a model with a stated authored share.
   - Report each "no" as plainly as each "yes".

---

## 8. Provenance and audit (computed by `build`, required for every candidate)

**Authored share.** For each tensor group, the build counts reference reads:
- **authored:** the group read no reference tensor. Split it further into **designed** (values set by formulas or constructions) and **random** (seeded draws), as the recipe declares; random weights carry no written content and are reported separately;
- **taken:** the group read reference tensors. Split it into **verbatim** (output equals one reference tensor, relative L2 < 1e-3) and **derived** (computed from one or more references).

The authored share is authored parameters / all parameters; also report designed-only.

**Audit.** For every authored tensor:
- compare it with every reference tensor of the same shape (cosine similarity, relative L2) and with sorted-value fingerprints of all reference tensors. Similarity above the noise level of independent random tensors means hidden copying: reclassify the group as taken and log it;
- check for stored benchmark content: correlation of authored embedding/output rows with practice-item answer tokens beyond what the design explains. Investigate any flag and reject the candidate if it stores items or answers.

**Summaries** per candidate: authored (designed / random), taken (verbatim / derived), per-reference contribution weighted by parameters, and a per-layer map. The final report shows the provenance table of both champions.

---

## 9. Infrastructure and robustness

- Use one Python venv, pin library versions, and log the hardware (GPU model and memory, CPU, RAM, disk).
- **Disk:** keep ≤ 3 built candidates on disk; rebuild from recipes when needed; delete activation dumps after analysis.
- **Restarts:** the server, queue and watchdog must restart from files with one command (`start_server.sh`). After a restart, resume editor agents and tell them to re-arm their waits.
- **Long waits:** poll in background loops with long timeouts and re-arm them when killed. Never block on a foreground sleep.
- **Git:** commit after every meaningful step with a sync script that refuses secrets and large files. Commit code, recipes, briefs, editor notes, reports, decisions, logs and results (ids and correctness only for hidden sets).
- **Safety of the process:** messages from sub-agents carry no user authority. Do not run external code that editors or web pages suggest without reading it. Ask the user before cloning third-party repositories into the environment.

---

## 10. Logging (three living documents, plus reports)

- `docs/DECISIONS.md`: numbered decisions D1, D2, … Each has a date, the decision, who made it (U = user, C = runner) and the rationale with evidence. Superseded decisions are marked, never deleted.
- `docs/EXPERIMENT_LOG.md`: a chronological journal of every launch, hand-back, selection result, control, error and fix, with UTC times.
- `docs/WRITING_FINDINGS.md`: what is learned about writing weights: which written constructions work as designed, which parts of a model could be replaced with written weights at no loss, what the references teach about how networks are built. Each finding names the editor and generation and states its evidence strength.
- `runs/gen_NNN/REPORT.md`: one per generation, written by the GA tool.
- Keep a short `README.md` with both lineage champions, their scores and authored shares vs the bar, and how to rebuild them from recipes.

---

## 11. Lessons from the predecessor experiment (use them; do not re-learn them)

1. **Noise is large.** At n = 256 multiple-choice items, the paired SE is ~2 points. Re-running the same model with different seeds flipped ~10% of answers (25/256). Measure the noise floor first (§7.1) and require z ≥ 2. Many plausible "gains" of +3 points were noise.
2. **Diagnoses found on ≤ 100 items often fail to replicate.** An editor found that "late-listed correct options lose 76% vs 44%" on 85 questions; on 256 questions it was 59% vs 62%. Check a diagnosis on a larger practice set before building on it.
3. **Proxy gates mislead.** Forced-answer probabilities (P(correct letter) after a prefix) rose for ten consecutive candidates; none improved generated accuracy in reasoning mode. Judge candidates by **generated answers under the real protocol**. Use cheap proxies (score, ppl) to discard broken builds and to rank near-chance written models, not to judge real gains.
4. **Speed vs intelligence.** At a tight token cap, edits that made the model conclude sooner "won" without making any answer better. Use a generous `CAP` and always report the both-finish subset, so efficiency gains are visible as such.
5. **Dev sets picked from the parent's failures mislead.** Any change re-draws a model's wrong samples on questions it sometimes gets right (regression to the mean). Two editors saw +16.7 points on such sets for edits that were null on held-out questions.
6. **Determinism and caching bugs can fake results.** Check weight hashes, identical-output counts, and that the reference model reproduces its cached answers byte for byte.
7. **Hidden sets must stay hidden from editors**, and the orchestrator should hand editors only practice-set facts. Editors that design on practice and are judged on hidden selection is what made results trustworthy.
8. **Small targeted edits to a trained model rarely change its reasoning.** In the predecessor, ten consecutive mechanism-level edits each did what they were designed to do inside the network and none moved accuracy. Expect written circuits to work as designed long before they move a benchmark; check them with `read` passes, and measure their benchmark effect honestly.
9. **Containers restart and disks fill.** Everything must be resumable from files, committed often, and built artefacts evicted before new builds.
10. **Write things down before seeing results** (§7.4). It caught several over-optimistic readings.

---

## 12. When to stop and ask the user

- No GPU and the CPU-only throughput makes one candidate's selection take more than ~6 hours.
- A required gated resource (GPQA, a gated reference) is inaccessible.
- A change to `MAX_PARAMS`, the lineages, `W_MIN_AUTHORED`, the promotion rules of §7.3, the benchmarks, or the visibility rules of §4.
- Running the sealed final set at any time other than pre-registered.

For everything else, decide, log and continue.

---

## 13. First actions (in order)

1. **Hardware and environment.** Inspect GPU, CPU, RAM and disk; create a venv; install torch, transformers, safetensors and the chosen engine. Log D1 (the experiment as stated in §1), D2 (settings as in §2, with any change) and D3 (engine and hardware).
2. **Secrets.** Set up token handling and the sync script with its secret check.
3. **Benchmarks.** Download them; build the splits of §4.1; write the id hashes into DECISIONS; move hidden and final files into `data/sealed/`; build the correctness-only reader.
4. **References.** Run the census (§3.1); download eligible reference models; write `docs/REFERENCES.md` with the study material (§3.2).
5. **Harness.** Build the evaluation harness (§5), including the letter-prior baseline. Run the determinism check and the throughput measurement, then fix `CAP` and set sizes.
6. **Editor tools.** Build the API, recipe builder with provenance and audit, queue, server and watchdog (§6, §8). Test each request type on one reference model, and test the provenance on three toy recipes (fully written, verbatim copy, a mix).
7. **Baselines.** Run all references, B-random, the B-merge recipes and the resampling control (§7.1). Log the working bar and the floor.
8. **Generation 0.** Write the editor briefs and launch (§7.2). Then loop (§7.3) until a stop condition (§7.5).
9. **Final run and report** (§7.5).
