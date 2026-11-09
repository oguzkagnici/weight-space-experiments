# Tweet / thread draft

## Single post (fits with a link)

Can AI agents make a model smarter by editing its weights by hand, no training?

Replicating Exp 8 of "Sparks of Weight-Space Programming" on a CPU, Claude agents edited 0.002% of LFM3-1.7B.

Held-out MMLU-Pro: 33.6% → 45.3%
GPQA Diamond: 12.6% → 19.2%

<link>

## Thread version

Paste each block as one post. The last post takes the link to the write-up (counts as 23 characters).

Can an AI make a model smarter by editing its weights directly, with no training?

We replicated Exp 8 of "Sparks of Weight-Space Programming" on a 4-core CPU. Claude agents edited 0.002% of LFM3-1.7B's weights:

MMLU-Pro (held out): 33.6% → 45.3%
GPQA Diamond: 12.6% → 19.2%

---

How: agents read the model's activations, wrote small gated "circuits" into MLP neurons by hand, and competed in a genetic loop. An edit was kept only if it beat the current model with z ≥ 2 on hidden questions. 2 of 9 generations produced a winner.

---

What the edits fix: the model thinks too long. It states the right answer, then reopens ("But let's double-check…") until it runs out of budget.

The 3 kept units close the thinking at a stated answer, force a commit between two finalists, and shorten the final answer.

---

Caveat: all numbers are at a 3,072-token thinking budget (the paper used 32k), so much of the gain is finishing in time. We're re-running base vs edited at 8,192 tokens now.

Full write-up, data and logs: <link>
