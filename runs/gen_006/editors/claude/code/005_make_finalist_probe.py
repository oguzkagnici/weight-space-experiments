# Hypothesis: at the paragraph end of a two-finalist statement, the parent's forced answer (thinking closed there)
# picks one of the two finalists, and picks gold clearly more often than chance when gold is among them; so committing
# at that point would rescue a good share of the 18 cut traces that narrow to two.
# Builds the read request for pass 1 (original = parent): texts = prompt + trace up to the end of the paragraph holding
# a two-finalist statement + "\n</think>\n\nThe answer is (".
import json, re
from wsp.editor import render_think, tok
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
L = r"\(?([A-J])\)?"
pair_pat = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + L + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + L + r"\b"
                      r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + L + r"\s+or\s+" + L + r"\b"
                      r"|\b" + L + r"\s+(?:or|vs\.?)\s+" + L + r"\s*[\?\.]", re.I)
SUFFIX = "\n</think>\n\nThe answer is ("
texts, meta = [], []
for qid, r in rows.items():
    raw = r["raw"]
    body = raw[len("<think>"):] if raw.startswith("<think>") else raw
    end_think = body.find("</think>")
    think = body if end_think < 0 else body[:end_think]
    for m in pair_pat.finditer(think):
        g = [x for x in m.groups() if x]
        if len(g) != 2 or g[0] == g[1]: continue
        cut_at = think.find("\n\n", m.end())
        if cut_at < 0: cut_at = len(think)
        text = render_think(qid) + think[:cut_at] + SUFFIX
        texts.append(text)
        meta.append({"id": qid, "cut": r["cut"], "correct": r["correct"], "gold": r["gold"], "pred": r["pred"],
                     "pair": sorted(g), "pos": m.start(), "frac": m.start() / len(think), "cut_at": cut_at,
                     "ntok": len(tok.encode(text, add_special_tokens=False))})
print(len(texts), "texts,", sum(x["ntok"] for x in meta), "tokens; max", max(x["ntok"] for x in meta))
json.dump(meta, open("work/probe1_meta.json", "w"), indent=0)
req = {"type": "read", "model": "original", "texts": texts,
       "return": {"hidden_states": [24], "positions": "last", "topk": 20}}
json.dump(req, open("work/req_01.json", "w"))

# Control (added before submission): forced answer at the end of each cut trace that narrowed to two (at the cap).
ctrl_ids = sorted({m["id"] for m in meta if m["cut"]})
for qid in ctrl_ids:
    r = rows[qid]; raw = r["raw"]; body = raw[len("<think>"):] if raw.startswith("<think>") else raw
    think = body.split("</think>")[0]
    cut_at = think.rfind("\n\n")
    text = render_think(qid) + think[:cut_at] + SUFFIX
    texts.append(text)
    meta.append({"id": qid, "cut": 1, "correct": 0, "gold": r["gold"], "pred": None, "pair": None, "control": "end",
                 "frac": cut_at / len(think), "cut_at": cut_at, "ntok": len(tok.encode(text, add_special_tokens=False))})
print(len(texts), "texts with controls,", sum(x["ntok"] for x in meta), "tokens")
json.dump(meta, open("work/probe1_meta.json", "w"), indent=0)
req["texts"] = texts
json.dump(req, open("work/req_01.json", "w"))
