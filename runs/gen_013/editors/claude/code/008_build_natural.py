# Pass 2 hypothesis: the corrupt-vs-clean detector tracks SELF-CONSISTENCY (surprise), not truth, so it does not fire at the parent's own wrong
# beliefs. Test: at the first sentence where a pass@4 sample states its pick, compare the model's own letter with a substituted letter
# (wrong samples: gold; right samples: the most common wrong pick of that question, else a fixed other letter). If the network knew its pick was
# wrong, the gold-substituted version would score LESS error-like than its own. Plus one hand-built natural step pair (9251 s0 lens sign).
import json, re, collections, random
from wsp.editor import *
random.seed(2)
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
byq = collections.defaultdict(list)
for r in pk: byq[r["id"]].append(r)
PAT = re.compile(r"(?:answer is|answer:|answer would be|answer likely|choose|pick|go with|select|best answer is|correct answer is)\s*\(?([A-J])\)?(?![\w])[^\n.]*[.\n]")
items = []
for r in pk:
    if r["cut"] or not r["pred"]: continue
    tt = trace_text(r); p0 = tt.index("<think>") + 7 if "<think>" in tt else 0; body = tt[p0:]
    m = next((m for m in PAT.finditer(body) if m.group(1) == r["pred"] and m.start() > 300), None)
    if not m: continue
    X = m.group(1); gold = r["gold"]
    if r["correct"]:
        wr = collections.Counter(x["pred"] for x in byq[r["id"]] if x["pred"] and x["pred"] != gold)
        Y = wr.most_common(1)[0][0] if wr else random.choice([c for c in "ABCDEFGHIJ" if c != gold])
    else: Y = gold
    end = p0 + m.end(); st = p0 + m.start(1)
    own = tt[:end]; sub = tt[:st] + Y + tt[st + 1: end]
    items.append(dict(id=r["id"], s=r["sample"], correct=r["correct"], X=X, Y=Y, own=own, sub=sub, frac=m.start() / len(body), sent=m.group(0)[:80]))
n = lambda t: len(tok(t, add_special_tokens=False).input_ids)
random.shuffle(items); sel, tot = [], 0
wr_first = sorted(items, key=lambda i: (i["correct"], random.random()))
cnt = collections.Counter()
for it in wr_first:
    k = 2 * n(it["own"])
    if k > 2 * 3600 or tot + k > 186000 or cnt[it["correct"]] >= 25: continue
    sel.append(it); tot += k; cnt[it["correct"]] += 1
# hand pair: 9251 s0 lens sign convention (model's own false step vs the corrected step)
r = next(x for x in pk if x["id"] == "mmlupro_9251" and x["sample"] == 0); tt = trace_text(r)
bad = "1/f = 1/v + 1/u = 1/16.6 + 1/(-50) = 1/16.6 - 1/50."
i = tt.index(bad) if bad in tt else -1
print("9251 sentence found at", i)
if i > 0:
    good = "1/f = 1/v - 1/u = 1/16.6 - 1/(-50) = 1/16.6 + 1/50."
    sel.append(dict(id="mmlupro_9251", s=0, correct=0, X="own-false-step", Y="corrected", own=tt[: i + len(bad)], sub=tt[:i] + good, frac=0, sent=bad, hand=1))
    tot += 2 * n(tt[: i + len(bad)])
print("items", len(items), "selected", len(sel), dict(cnt), "tokens", tot, "mean frac", round(sum(s["frac"] for s in sel) / len(sel), 2))
texts = []
for it in sel: texts += [it["own"], it["sub"]]
json.dump(sel, open("work/natural2.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"token_logprobs": True, "topk": 50, "positions": "last", "hidden_states": [8, 14, 18, 20, 24]}}, open("work/req_read2.json", "w"))
for it in sel[:4]: print(it["id"], it["s"], it["correct"], it["X"], "->", it["Y"], "|", it["sent"].encode("ascii", "replace").decode())
