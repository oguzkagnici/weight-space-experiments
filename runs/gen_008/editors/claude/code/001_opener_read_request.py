# Pass 1 request (read, edited = carried stack). Hypothesis: at a commit opener ("Thus", "So") that starts the paragraph
# right after a two-finalist statement, the L23 MLP input carries a separable "about to decide between two finalists"
# state (two-finalist context + commit-opener token), distinct from commit openers elsewhere; and the model's next token
# there is mostly a deliberation continuation (" we", " maybe", " let", " examine"), not a decision (" I", " the answer").
import json, re, random
from wsp.editor import tok, trace_text, render_think
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
R = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
g = json.load(open("/home/user/weight-space-experiments/runs/gen_007/editors/claude/work/groups.json"))
X = r"\(?([A-J])\)?"
pair = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + X + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + X + r"\b"
                  r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + X + r"\s+or\s+" + X + r"\b"
                  r"|\b" + X + r"\s+(?:or|vs\.?)\s+" + X + r"\s*[\?\.]", re.I)
texts, meta = [], []
for q in g["narrow"] + g["fin_narrow"]:
    full = trace_text(R[q]); p = render_think(q); t = full[len(p):]
    for m in pair.finditer(t.split("</think>")[0]):
        if len({x for x in m.groups() if x}) != 2: continue
        e = t.find("\n\n", m.end())
        if e < 0: continue
        for op in ("Thus", "So"):
            texts.append(p + t[:e + 2] + op); meta.append({"id": q, "opener": op, "kind": "pos", "cut_char": e + 2})
        break                                              # first two-finalist paragraph only
others = sorted(q for q in g["others"] if not R[q]["cut"])
random.Random(8).shuffle(others)
for q in others[:10]:
    texts.append(trace_text(R[q])); meta.append({"id": q, "kind": "full_finished"})
ntok = sum(len(tok(x, add_special_tokens=False)["input_ids"]) for x in texts)
print(len(texts), "texts,", ntok, "tokens")
json.dump(meta, open("work/p1_meta.json", "w"), indent=0)
json.dump({"type": "read", "model": "edited", "texts": texts,
           "return": {"token_logprobs": True, "topk": 20, "inputs": ["model.layers.23.feed_forward"], "positions": "all"}},
          open("work/req_01.json", "w"))
