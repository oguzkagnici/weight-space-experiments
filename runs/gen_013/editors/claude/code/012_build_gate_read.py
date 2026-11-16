# Gate read (pass 3, EDITED = L23 n881 detector->re-check unit, K 4, ALPHA 1, PCT 85): on HELD-OUT pass-1 pairs (test questions, not used to fit)
# measure P(re-check next) corrupt vs clean (parent values = pass 1, same texts) and per-token logprob changes (off-target); plus natural
# own-vs-substituted pairs from pass 2 (does the unit raise re-checking after the GOLD substitution / corrected step, i.e. doubt true steps?).
import json
from wsp.editor import *
P = json.load(open("work/pairs1.json")); trq = set(json.load(open("work/train_qs.json")))
N = json.load(open("work/natural2.json"))
n = lambda t: len(tok(t, add_special_tokens=False).input_ids)
texts, index, tot = [], [], 0
for i, p in enumerate(P):
    if p["id"] in trq: continue
    texts += [p["clean"], p["corrupt"]]; index += [("p1", 2 * i), ("p1", 2 * i + 1)]; tot += n(p["clean"]) + n(p["corrupt"])
for j, it in sorted(enumerate(N), key=lambda x: (not x[1].get("hand"), n(x[1]["own"]))):
    k = 2 * n(it["own"])
    if tot + k > 199000: continue
    texts += [it["own"], it["sub"]]; index += [("p2", 2 * j), ("p2", 2 * j + 1)]; tot += k
print("texts", len(texts), "tokens", tot, "p1 pairs", sum(1 for x in index if x[0] == "p1") // 2, "p2 pairs", sum(1 for x in index if x[0] == "p2") // 2)
json.dump(index, open("work/read3_index.json", "w"))
json.dump({"type": "read", "model": "edited", "texts": texts, "return": {"token_logprobs": True, "topk": 50, "positions": "last", "hidden_states": [22]}}, open("work/req_read3.json", "w"))
