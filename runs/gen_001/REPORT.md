# Generation 1 report

- Created: 2026-11-04 19:30 UTC
- Parent: `base`
- Editors: claude, ed_a, ed_b; budget 10 passes each; up to 2 candidates each
- Hidden selection subset: 600 questions (seed `selection-gen-1`)

## Suggested starting points

- **ed_a**: Answer-letter preference. The model may prefer some option letters (positions) regardless of content; measure this on practice questions (predicted vs true letter counts, and the letter probabilities on questions it gets wrong). If it is there, find what writes the position preference into the answer position (heads, MLP neurons, a constant bias direction, or a dependence on where each option sits in the prompt) and remove it in a content-preserving way. A uniform prior shift tuned to 10 options may not transfer to 4-option GPQA, so prefer fixing the mechanism.
- **ed_b**: Near misses and late-layer loss. When the model is wrong, check how often the correct letter is its second choice. Use the logit lens (residual stream after each layer, through the final norm and the tied output layer, restricted to the option letters) to find whether the correct answer leads at some middle layer and is lost later, and which late components (MLP neurons, attention heads) overwrite it.
- **claude**: How the answer is routed from the option text to the answer position: which attention heads at the last token attend to the option lines, whether they select by content or by position, and whether some heads systematically move the wrong option.

## Forward passes

| editor | pass | type | model | n | headline | seconds |
|---|---|---|---|---|---|---|
| ed_b | 1 | score +acts | original | 512 | acc 36.72 | 622.4 |

## Candidates

## Editors' reports

### claude

_(no report yet)_

### ed_a

_(no report yet)_

### ed_b

_(no report yet)_

