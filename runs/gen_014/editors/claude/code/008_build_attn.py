# Hypothesis: at option-judging positions, right samples attend more to the question stem (and to the decisive cue) than wrong samples of the same question; build attn_last read on prefixes.
import json, re, sys, collections
sys.path.insert(0, "code")
from wsp.editor import *
from importlib import import_module
CUES = import_module("005_cues").CUES
Q = questions()
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
MIXED = ["mmlupro_10964","mmlupro_1171","mmlupro_2055","mmlupro_2654","mmlupro_2731","mmlupro_5325","mmlupro_6110","mmlupro_6562","mmlupro_6800","mmlupro_7456"]
by = collections.defaultdict(list)
for r in pk: by[r["id"]].append(r)
def judge_pos(think, L, start):
    pat = re.compile(r"Option\s+%s\b|\(%s\)|(?m:^\s*[-*]?\s*%s[\):])" % (L, L, L))
    for m in pat.finditer(think):
        if m.start() >= start: return m.end()
    return None
texts, meta = [], []
for q in MIXED:
    gold = Q[q]["answer"]
    wrongs = collections.Counter(r["pred"] for r in by[q] if not r["cut"] and not r["correct"])
    fin = wrongs.most_common(1)[0][0]
    for r in by[q]:
        if r["cut"]: continue
        think = r["raw"][len("<think>"):].split("</think>")[0]
        L = len(think); pts = {}
        pts["j_gold"] = judge_pos(think, gold, int(0.1 * L))
        pts["j_fin"] = judge_pos(think, fin, int(0.1 * L))
        # commit paragraph start: first commit to own pick in last half
        pat = re.compile(r"(answer|choose|pick|go with|select|correct)[^\n]{0,40}\(?\b%s\b\)?" % r["pred"], re.I)
        b = [m.end() for m in re.finditer(r"\n\n", think)]
        for m in pat.finditer(think):
            if m.start() > 0.5 * L:
                prev = [x for x in b if x <= m.start()]; pts["j_late"] = prev[-1] if prev else None; break
        for k, c in pts.items():
            if c is None: continue
            texts.append(render_think(q) + think[:c])
            meta.append(dict(id=q, sample=r["sample"], correct=r["correct"], pred=r["pred"], gold=gold, fin=fin, point=k, frac=c / L))
tot = sum(len(tok.encode(t)) for t in texts)
print("texts", len(texts), "tokens", tot, collections.Counter((m["point"], m["correct"]) for m in meta))
json.dump(meta, open("work/p2_meta.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"positions": "last", "attn_last": [2,4,6,8,10,12,14,16,18,20,22]}}, open("work/req_p2.json", "w"))
for m, t in zip(meta, texts):
    if m["id"] == "mmlupro_6800" and m["sample"] in (0, 1): print(m["point"], m["correct"], round(m["frac"], 2), repr(t[-120:].encode("ascii","replace").decode()))
# per-text limit 4,096 tokens: drop longer prefixes
n = [len(tok.encode(t)) for t in texts]
sel = [i for i, k in enumerate(n) if k <= 4090]
texts = [texts[i] for i in sel]; meta = [meta[i] for i in sel]
print("after limit:", len(texts), "texts", sum(n[i] for i in sel), "tokens", collections.Counter((m["point"], m["correct"]) for m in meta))
json.dump(meta, open("work/p2_meta.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"positions": "last", "attn_last": [2,4,6,8,10,12,14,16,18,20,22]}}, open("work/req_p2.json", "w"))
