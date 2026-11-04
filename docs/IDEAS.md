# Ideas for later (not implemented)

From the user, 2026-11-04 (during generation 2). Recorded only; the current runs are unchanged.

1. **Guard benchmarks against single-benchmark overfitting.**
   - Optimising one benchmark (MMLU-Pro) may silently cost performance elsewhere.
   - At each generation, also evaluate the champion and the candidates on a panel of other benchmarks, such as math, coding/SWE, computer use, instruction following and GPQA.
   - Require that the panel stays the same or improves. This could be a hard constraint (no regression beyond a tolerance) or part of a multi-objective fitness.
   - Current state: GPQA Diamond is evaluated at checkpoints only. Generative and agentic benchmarks need GPU-scale decoding (decision D5).

2. **Turn the loop into a reusable environment.**
   - It would take a model and one or more benchmarks and run the genetic weight-editing loop.
   - It would include editors, a forward-pass tool, hidden selection, guard benchmarks and logging.
   - What is already generic in `wsp/`: the GA driver, the server queue, budgets, briefs and reports.
   - What is model- or benchmark-specific today:
     - `config.py` (layer types, forbidden tensors);
     - `data.py` (MMLU-Pro and GPQA loaders, the prompt);
     - `engine.py` (direct-answer letter scoring, hook names);
     - the brief's architecture paragraph.
   - These would become adapters: a model adapter (tensor names, forbidden list, hook points, architecture description) and a benchmark adapter (loader, split, prompt rendering, scorer, and the generative or likelihood protocol).

Observations from the runs that bear on these ideas:
- Generation 1 showed that accuracy and calibration (log-probability of the correct answer) can move in opposite directions. A guard panel should track both.
- Editors' own holdout estimates were inflated by choosing among variants. A reusable environment should keep the hidden selection subset and the "choose on DEV, confirm once" protocol.
