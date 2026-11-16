# Pass 4: record the L23 MLP w1 inputs at ALL positions (original model; identical under the edit, which acts only at the L23 MLP output)
# on (a) 12 held-out pass-1 pairs (test questions) and (b) 8 practice trace-bank traces (8,192 bank, not pass@4), truncated to 2,600 tokens,
# to refit the n881 gate against real background positions (incl. prompt/sink tokens).
import json, random
from wsp.editor import *
random.seed(4)
P = json.load(open("work/pairs1.json")); trq = set(json.load(open("work/train_qs.json")))
n = lambda t: len(tok(t, add_special_tokens=False).input_ids)
te = [i for i, p in enumerate(P) if p["id"] not in trq]; random.shuffle(te)
texts, index, tot = [], [], 0
for i in te:
    p = P[i]; k = n(p["clean"]) + n(p["corrupt"])
    if k > 2400 or len([x for x in index if x["kind"] == "pair"]) >= 24: continue
    texts += [p["clean"], p["corrupt"]]; index += [dict(kind="pair", i=i, v="clean"), dict(kind="pair", i=i, v="corrupt")]; tot += k
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pkq = {json.loads(l)["id"] for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")}
cand = [r for r in bank if r["id"] not in pkq and r["id"] not in {P[i]["id"] for i in range(len(P))}]; random.shuffle(cand)
for r in cand[:8]:
    t = trace_text(r); ids_ = tok(t, add_special_tokens=False).input_ids[:2600]; t = tok.decode(ids_)
    texts.append(t); index.append(dict(kind="bank", id=r["id"], correct=r["correct"])); tot += len(ids_)
print("texts", len(texts), "tokens", tot, "pairs", sum(1 for x in index if x["kind"] == "pair") // 2, "MB float32", round(tot * 2048 * 4 / 1e6))
json.dump(index, open("work/read4_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"token_logprobs": True, "positions": "all", "inputs": ["model.layers.23.feed_forward.w1"]}}, open("work/req_read4.json", "w"))
