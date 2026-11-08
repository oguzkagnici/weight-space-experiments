# Pass 2 request (read, parent): same 39 texts as pass 1 (labels from code/006 reused). Record the L23 MLP input (gate
# building), the L23 MLP output (to estimate the post_ffn_norm effect of a new unit) and h24 (final residual, for the
# logit read-out through the final norm). Hypothesis: a gated L23 unit on a "conclusion just stated" direction can
# raise P(</think>) at statement ends from ~0 to > 0.5 with almost no change elsewhere.
import json
texts = json.load(open("work/req_01.json"))["texts"]
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"inputs": ["model.layers.23.feed_forward"], "outputs": ["model.layers.23.feed_forward"],
                      "hidden_states": [24], "positions": "all"}}, open("work/req_02.json", "w"))
print(len(texts), "texts")
