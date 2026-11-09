# Tweet / thread draft

Numbers are held-out MMLU-Pro (128 questions) unless stated; every number names its thinking budget (D61).

## Single post (fits with a link)

Can AI agents make a model smarter by editing its weights by hand, no training?

Replicating Exp 8 of "Sparks of Weight-Space Programming", Claude agents edited 0.002% of LFM3-1.7B.

Held-out MMLU-Pro:
3k thinking tokens: 33.6% → 45.3%
8k: 56.3% → 64.1%

<link>

## Thread version

Paste each block as one post. The last post takes the link to the write-up (counts as 23 characters).

Can an AI make a model smarter by editing its weights directly, with no training?

We replicated Exp 8 of "Sparks of Weight-Space Programming" on a 4-core CPU. Claude agents edited 0.002% of LFM3-1.7B's weights.

Held-out MMLU-Pro: 33.6% → 45.3% (3k-token thinking budget)

---

How: agents read the model's activations, wrote small gated "circuits" into MLP neurons by hand, and competed in a genetic loop. An edit was kept only if it beat the current model with z ≥ 2 on hidden questions. 2 of 9 generations produced a winner.

---

What the edits fix: the model thinks too long. It states the right answer, then reopens ("But let's double-check…") until it runs out of budget.

The 3 kept units close the thinking at a stated answer, force a commit between two finalists, and shorten the final answer.

---

Is it just finishing faster? We re-ran both at 8k tokens: 56.3% → 64.1%. The edited model leads at every budget from 512 to 8k and matches the original's 8k score at ~4.8k.

GPQA Diamond (3k, never used for tuning): 12.6% → 19.2%.

Write-up and logs: <link>
