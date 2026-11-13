# Hypothesis (replication of gen 10): in finished traces whose last pair statement contains gold and the pick, the pick is the first-named letter far more often than the second-named; the model names its favourite first.
import json, collections
P = json.load(open("work/pairs_all.json"))
last = {}
for p in P: last[(p["id"], p["sample"], p["src"])] = p
c = collections.Counter()
for k, p in last.items():
    if p["cut"] or p["pred"] is None: continue
    if p["gold"] in (p["a"], p["b"]) and p["pred"] in (p["a"], p["b"]):
        gold_first = p["gold"] == p["a"]; pick_first = p["pred"] == p["a"]
        c[(("gold1st" if gold_first else "gold2nd"), ("pick1st" if pick_first else "pick2nd"))] += 1
    elif p["pred"] in (p["a"], p["b"]):
        c[("gold_out", "pick1st" if p["pred"] == p["a"] else "pick2nd")] += 1
for k in sorted(c): print(k, c[k])
