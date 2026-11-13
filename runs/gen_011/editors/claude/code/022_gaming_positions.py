# H22 (pass 4 design): positions where the model is about to write the gaming noun ("exam/test/questions/maker/MCQ" in a gaming phrase), within the
# first 4,090 tokens; negatives = positions before " exam"/" test" used in CONTENT (not in a gaming phrase) + all other positions.
import json, re, collections
from wsp.editor import *
G = re.compile(r"(typical (exam|test|mcq|multiple)|in such (questions|exams)|test[- ]?maker|exam (style|question)s? (often|usually|typically|would|likely|expect)|(exam|question|test) (likely|probably) expects?)", re.I)
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
b3 = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl")]
pk = {(r["id"], r["sample"]) for r in rows}; rows += [r for r in bank if (r["id"], 0) not in pk]
rows3 = b3
EX = {1423, 1426}   # " exam", " test"
items = []; tot = 0
for src, R in (("8192", rows), ("3072", rows3)):
    for r in R:
        full = trace_text(r); pr = render_think(r["id"])
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True); ids = enc.input_ids[:4090]; offs = enc.offset_mapping[:4090]
        pl = len(tok(pr, add_special_tokens=False).input_ids)
        txt = full; pos, neg = [], []
        for m in G.finditer(txt):
            # the gaming noun = second word of the phrase; position = token BEFORE the token that starts it
            w = re.search(r"(exam|test|mcq|multiple|questions|exams|maker|style|question)", m.group(0)[1:], re.I)
            c = m.start() + 1 + w.start() if w else m.start()
            j = next((j for j in range(pl, len(ids)) if offs[j][0] >= c - 1 and offs[j][1] > c), None)
            if j is not None and j > pl: pos.append(j - 1)
        gs = set(range(len(ids)))
        for j in range(pl, len(ids)):
            if ids[j] in EX and (j - 1) not in pos: neg.append(j - 1)
        if pos or (neg and len(items) % 3 == 0):
            items.append({"src": src, "id": r["id"], "sample": r.get("sample", 0), "n": len(ids), "plen": pl, "pos": pos, "neg_content": neg, "correct": r["correct"]})
items.sort(key=lambda x: -len(x["pos"]))
sel = []; tot = 0
for it in items:
    if tot + it["n"] > 140000: continue
    sel.append(it); tot += it["n"]
print("items with positives", sum(1 for i in items if i["pos"]), "selected", len(sel), "tokens", tot,
      "positives", sum(len(i["pos"]) for i in sel), "content-negatives", sum(len(i["neg_content"]) for i in sel))
texts = []
for it in sel:
    R = rows if it["src"] == "8192" else rows3
    r = next(x for x in R if x["id"] == it["id"] and x.get("sample", 0) == it["sample"])
    ids = tok(trace_text(r), add_special_tokens=False).input_ids[:4090]; texts.append(tok.decode(ids))
json.dump(sel, open("work/read4_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"inputs": ["model.layers.23.feed_forward.w1"], "positions": "all", "topk": 10}}, open("work/req_read4.json", "w"))
