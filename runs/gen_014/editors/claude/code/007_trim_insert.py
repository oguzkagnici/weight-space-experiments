# Hypothesis: (budget) keep the insertion test inside 200k tokens: early cut x 3 variants on all finished-wrong samples; late cut x cue/ctrl on one wrong sample per question.
import json
from wsp.editor import *
meta = json.load(open("work/insert_meta.json")); req = json.load(open("work/req_p1.json"))
keep, seen = [], set()
first_wrong = {}
for m in meta:
    if not m["correct"] and m["id"] not in first_wrong: first_wrong[m["id"]] = m["sample"]
for i, m in enumerate(meta):
    if m["correct"]: continue
    if m["cut"] == "early": keep.append(i)
    elif m["variant"] != "none" and m["sample"] == first_wrong[m["id"]]: keep.append(i)
texts = [req["texts"][i] for i in keep]; meta2 = [meta[i] for i in keep]
tot = sum(len(tok.encode(t)) for t in texts)
print("texts", len(texts), "tokens", tot)
json.dump(meta2, open("work/p1_meta.json", "w"))
req["texts"] = texts; json.dump(req, open("work/req_p1.json", "w"))
# per-text limit 4,096: drop pairs (cue/ctrl of the same sample and cut) where either text is longer
n = [len(tok.encode(t)) for t in texts]
bad = {(m["id"], m["sample"], m["cut"]) for m, k in zip(meta2, n) if k > 4090}
sel = [i for i, m in enumerate(meta2) if (m["id"], m["sample"], m["cut"]) not in bad]
texts = [texts[i] for i in sel]; meta2 = [meta2[i] for i in sel]
print("after 4,096 limit: texts", len(texts), "tokens", sum(n[i] for i in sel), "dropped", sorted(bad))
json.dump(meta2, open("work/p1_meta.json", "w")); req["texts"] = texts; json.dump(req, open("work/req_p1.json", "w"))
