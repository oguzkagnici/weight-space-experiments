# Pass 1 request (read, original = parent). Hypothesis: at hedged in-think answer statements ("Thus likely answer is (E).")
# the parent's L20 n5414 wrap-up gate stays below threshold, while at firm final statements it fires; the L20 MLP input
# there separates "hedged statement just made" from other positions, which a second gated unit could use.
import json, re, random
from wsp.editor import tok, trace_text, render_think
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"
R = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
ans = re.compile(r"answer is:?\s*\(?\*?\*?([A-J])\b\)?")
texts, meta = [], []
for q, r in sorted(R.items()):
    ms = list(ans.finditer(r["think"]))
    if not ms: continue
    first = ms[0]; frac = first.start() / len(r["think"])
    full = trace_text(r); p = render_think(q); off = len(p) - len("<think>")   # think starts with "<think>"
    assert full[off:off + 200] == r["raw"][:200]
    if r["cut"] or frac < 0.9:            # mid-trace statement (hedged group)
        end = min(len(r["think"]), first.end() + 600)
        texts.append(full[:off + end]); meta.append({"id": q, "kind": "mid", "cut": r["cut"], "stmt_char": off + first.start(), "stmt_end": off + first.end()})
fins = sorted(q for q, r in R.items() if not r["cut"] and any(m.start() / len(r["think"]) >= 0.9 for m in ans.finditer(r["think"])))
random.Random(9).shuffle(fins)
for q in fins[:15]:
    r = R[q]; full = trace_text(r); off = len(render_think(q)) - len("<think>"); ms = list(ans.finditer(r["think"]))
    texts.append(full); meta.append({"id": q, "kind": "final_full", "cut": 0, "stmt_char": off + ms[-1].start(), "stmt_end": off + ms[-1].end()})
ntok = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in texts]
print(len(texts), "texts", sum(ntok), "tokens", max(ntok), "max", {k: sum(m["kind"] == k for m in meta) for k in ("mid", "final_full")})
json.dump(meta, open("work/p1_meta.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"inputs": ["model.layers.20.feed_forward"], "positions": "all", "token_logprobs": True, "topk": 20}},
          open("work/req_01.json", "w"))
